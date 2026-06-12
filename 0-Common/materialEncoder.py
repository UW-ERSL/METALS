from networks import VariationalAutoencoder
import torch
from torch.func import vmap, jacrev
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from InterpolationFunctions import bezierInterpolation, bezierInterpolation_torch, TMin, TMax
from InterpolationFunctions import logBezierInterpolation, logBezierInterpolation_torch
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from scipy.stats import spearmanr
import os
import sys


class MaterialEncoder:
    def __init__(self, vae_params):
        self.nAttributes = 0
        self.vae_params = vae_params

    def readExcel(self, excel_file):
        self.excel_file = excel_file
        self.preprocessData()

        self.nMaterials = self.scaledMaterialData.shape[0]
        self.nAttributes = self.scaledMaterialData.shape[1]

        self.vaeSettings = {
            'encoder': {
                'inputDim': self.nAttributes,
                'hiddenDim': self.vae_params.vae_hiddenDim,
                'latentDim': self.vae_params.latentDim
            },
            'decoder': {
                'latentDim': self.vae_params.latentDim,
                'hiddenDim': self.vae_params.vae_hiddenDim,
                'outputDim': self.nAttributes
            }
        }
        self.vaeNet = VariationalAutoencoder(self.vaeSettings)

    def preprocessData(self):
        df = pd.read_excel(self.excel_file, header=None)
        attribute_names = df.iloc[0, 1:].tolist()
        units = df.iloc[1, 1:].tolist()
        material_names = df.iloc[2:, 0].tolist()
        values = df.iloc[2:, 1:].to_numpy(dtype=float)

        self.rawData = values

        normalizedData = np.zeros_like(values)
        dataScaleMin = np.zeros(values.shape[1])
        dataScaleMax = np.zeros(values.shape[1])

        for i, name in enumerate(attribute_names):
            col = values[:, i]
            if name in ['E0', 'E1', 'E2', 'E3', 'Y0', 'Y1', 'Y2', 'Y3']:
                log_col = np.log10(col)
                dataScaleMin[i] = log_col.min()
                dataScaleMax[i] = log_col.max()
                normalizedData[:, i] = (log_col - dataScaleMin[i]) / (dataScaleMax[i] - dataScaleMin[i] + 1e-12)
            else:
                norm_col = col
                dataScaleMin[i] = norm_col.min()
                dataScaleMax[i] = norm_col.max()
                normalizedData[:, i] = (norm_col - dataScaleMin[i]) / (dataScaleMax[i] - dataScaleMin[i] + 1e-12)

        scaledMaterialData = torch.tensor(normalizedData).float()

        materialAttributes = {}
        for i, name in enumerate(attribute_names):
            materialAttributes[name] = {
                'idx': i,
                'unit': units[i],
                'scaleMin': dataScaleMin[i],
                'scaleMax': dataScaleMax[i],
            }

        self.materialNames = material_names
        self.scaledMaterialData = scaledMaterialData
        self.materialAttributes = materialAttributes

        attribute_means = normalizedData.mean(axis=0)
        attribute_stds = normalizedData.std(axis=0)
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds
        self.computeParticipationRatio()
        print(f"Intrinsic dimension estimate (participation ratio): {self.intrinsic_dim:.3f}")
        print(f"PCA explained variance ratio: {self.pca_explained_variance_ratio}")
    def computeParticipationRatio(self):
        """
        Effective intrinsic dimensionality estimate from PCA eigenvalues.
        Uses normalized data already created in preprocessData.
        """
        X = self.scaledMaterialData.detach().cpu().numpy()
        pca = PCA().fit(X)
        lambdas = np.asarray(pca.explained_variance_, dtype=float)

        if np.all(lambdas <= 1e-16):
            self.intrinsic_dim = 0.0
            self.pca_explained_variance = lambdas
            self.pca_explained_variance_ratio = np.zeros_like(lambdas)
            return self.intrinsic_dim

        self.intrinsic_dim = float((np.sum(lambdas) ** 2) / np.sum(lambdas ** 2))
        self.pca_explained_variance = lambdas
        self.pca_explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
        return self.intrinsic_dim


    def computeLatentVarianceRatio(self):
        """
        Post-training check for dead latent directions.
        ratio = min Var(z_i) / max Var(z_i)
        """
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData).detach().cpu().numpy()

        if z_real.ndim == 1:
            z_real = z_real.reshape(-1, 1)

        var_per_dim = np.var(z_real, axis=0)
        vmax = np.max(var_per_dim)

        if vmax < 1e-16:
            ratio = 0.0
        else:
            ratio = float(np.min(var_per_dim) / vmax)

        return {
            "latent_variances": var_per_dim,
            "latent_variance_ratio": ratio,
        }


    def checkJacobianSpectrum(self, svd_tol=1e-8, verbose=True):
        """
        Better than raw rank: inspect singular values of decoder Jacobian at real material points.
        """
        z_real = self.vaeNet.encoder(self.scaledMaterialData).detach()
        latent_dim = self.vae_params.latentDim

        summary = {
            "ranks": [],
            "min_singular_values": [],
            "condition_numbers": [],
            "all_singular_values": [],
        }

        for zm in z_real:
            zm = zm.unsqueeze(0).clone().detach().requires_grad_(True)

            def decoder_single(zbatch):
                return self.vaeNet.decoder(zbatch).squeeze(0)

            J = torch.autograd.functional.jacobian(decoder_single, zm).squeeze(1)  # (nAttr, latentDim)
            if J.ndim == 1:
                J = J.reshape(-1, 1)

            svals = torch.linalg.svdvals(J).detach().cpu().numpy()
            rank = int(np.sum(svals > svd_tol))
            smin = float(np.min(svals))
            smax = float(np.max(svals))
            cond = float(smax / max(smin, svd_tol))

            summary["ranks"].append(rank)
            summary["min_singular_values"].append(smin)
            summary["condition_numbers"].append(cond)
            summary["all_singular_values"].append(svals)

        if verbose:
            print("\n" + "=" * 70)
            print("Decoder Jacobian spectrum at real material points")
            print("=" * 70)
            for i, name in enumerate(self.materialNames):
                print(
                    f"{name:20s} | rank={summary['ranks'][i]:2d} | "
                    f"smin={summary['min_singular_values'][i]:.3e} | "
                    f"cond={summary['condition_numbers'][i]:.3e}"
                )
            if min(summary["ranks"]) < latent_dim:
                print("WARNING: Rank-deficient Jacobian detected at one or more material points.")
            print("=" * 70 + "\n")

        return summary


    def checkPropertyOrdering(self, rho_warn=0.8, verbose=True):
        """
        Spearman rank check between raw material ordering and decoded ordering
        at the encoded material points.
        """
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            props = self.getMaterialProperties(self.vaeNet.decoder(z_real))

        results = {}
        for name, attr in self.materialAttributes.items():
            raw_vals = np.asarray(self.rawData[:, attr['idx']], dtype=float)
            dec_vals = props[name].detach().cpu().numpy()
            rho, pval = spearmanr(raw_vals, dec_vals)
            results[name] = {"rho": float(rho), "pval": float(pval)}

            if verbose:
                flag = " <-- WARNING: possible folding/order violation" if rho < rho_warn else ""
                print(f"{name:20s} | Spearman rho = {rho:.3f}{flag}")

        return results


    def interpolationDiagnostics(self, n_interp=101, verbose=True):
        """
        Check pairwise latent interpolations between encoded real materials.
        Reports overshoot and monotonicity violations property-by-property.
        """
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData).detach().cpu().numpy()

        if z_real.ndim == 1:
            z_real = z_real.reshape(-1, 1)

        attr_names = list(self.materialAttributes.keys())
        diagnostics = {name: {"overshoot_count": 0, "nonmonotone_count": 0} for name in attr_names}

        for i in range(len(self.materialNames)):
            for j in range(i + 1, len(self.materialNames)):
                t = np.linspace(0.0, 1.0, n_interp)
                z_line = (1.0 - t)[:, None] * z_real[i][None, :] + t[:, None] * z_real[j][None, :]
                z_line_t = torch.tensor(z_line, dtype=torch.float32)

                with torch.no_grad():
                    props_line = self.getMaterialProperties(self.vaeNet.decoder(z_line_t))

                for name, attr in self.materialAttributes.items():
                    vals = props_line[name].detach().cpu().numpy()
                    a = float(self.rawData[i, attr["idx"]])
                    b = float(self.rawData[j, attr["idx"]])
                    lo, hi = min(a, b), max(a, b)

                    # overshoot
                    if np.any(vals < lo - 1e-12) or np.any(vals > hi + 1e-12):
                        diagnostics[name]["overshoot_count"] += 1

                    # monotonicity
                    diffs = np.diff(vals)
                    sgn = np.sign(b - a)
                    if sgn > 0 and np.any(diffs < -1e-10):
                        diagnostics[name]["nonmonotone_count"] += 1
                    elif sgn < 0 and np.any(diffs > 1e-10):
                        diagnostics[name]["nonmonotone_count"] += 1

        if verbose:
            print("\n" + "=" * 70)
            print("Interpolation diagnostics")
            print("=" * 70)
            for name in attr_names:
                print(
                    f"{name:20s} | overshoot={diagnostics[name]['overshoot_count']:3d} | "
                    f"nonmonotone={diagnostics[name]['nonmonotone_count']:3d}"
                )
            print("=" * 70 + "\n")

        return diagnostics


    def computeLocalJacobianVariation(self, n_pairs=64, eps=1e-2, seed=0):
        """
        Cheap Hessian proxy: sample latent points near real materials and estimate
        ||J(z+dz) - J(z)|| / ||dz||.
        This is much cheaper and more scalable than full-grid Hessian norms.
        """
        rng = np.random.default_rng(seed)

        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData).detach().cpu().numpy()

        if z_real.ndim == 1:
            z_real = z_real.reshape(-1, 1)

        def decoder_single(zbatch):
            return self.vaeNet.decoder(zbatch).squeeze(0)

        vals = []

        for _ in range(n_pairs):
            k = rng.integers(0, z_real.shape[0])
            z = z_real[k].copy()
            dz = rng.normal(size=z.shape)
            ndz = np.linalg.norm(dz)
            if ndz < 1e-14:
                continue
            dz = eps * dz / ndz

            z1 = torch.tensor(z[None, :], dtype=torch.float32, requires_grad=True)
            z2 = torch.tensor((z + dz)[None, :], dtype=torch.float32, requires_grad=True)

            J1 = torch.autograd.functional.jacobian(decoder_single, z1).squeeze(1)
            J2 = torch.autograd.functional.jacobian(decoder_single, z2).squeeze(1)

            if J1.ndim == 1:
                J1 = J1.reshape(-1, 1)
                J2 = J2.reshape(-1, 1)

            num = torch.linalg.norm(J2 - J1).item()
            den = np.linalg.norm(dz)
            vals.append(num / max(den, 1e-14))

        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return {"mean_jacobian_variation": np.nan, "max_jacobian_variation": np.nan}

        return {
            "mean_jacobian_variation": float(np.mean(vals)),
            "max_jacobian_variation": float(np.max(vals)),
        }


    def latentDimensionSweepBetter(
        self,
        dim_range=range(1, 6),
        numEpochs=30000,
        klFactor=1e-3,
        learningRate=1e-3,
        hiddenDim=None,
        save_prefix="vae_dim",
    ):
        """
        Fresh-model latent-dimension sweep.
        Safer than mutating the current model in place.
        """
        results = {}

        for l in dim_range:
            # make a fresh shallow copy of settings
            self.vae_params.latentDim = l
            if hiddenDim is not None:
                self.vae_params.vae_hiddenDim = hiddenDim

            # rebuild model from data
            self.readExcel(self.excel_file)
            save_name = f"{save_prefix}_{l}.pt"

            self.trainAutoencoder(
                numEpochs=numEpochs,
                klFactor=klFactor,
                savedNet=save_name,
                learningRate=learningRate,
                maxAttributeErrorPercent=self.vae_params.maxAttributeErrorPercent,
            )

            recon_err = self.largestEncodingErrorPercent()
            latent_var = self.computeLatentVarianceRatio()
            order_diag = self.checkPropertyOrdering(verbose=False)
            jac_spec = self.checkJacobianSpectrum(verbose=False)

            results[l] = {
                "maxReconErrPct": float(recon_err),
                "latentVarianceRatio": float(latent_var["latent_variance_ratio"]),
                "minJacRank": int(min(jac_spec["ranks"])),
                "minSingVal": float(min(jac_spec["min_singular_values"])),
                "orderingMinRho": float(min(v["rho"] for v in order_diag.values())),
            }

            print(
                f"latentDim={l} | "
                f"maxReconErr={results[l]['maxReconErrPct']:.4f}% | "
                f"latentVarRatio={results[l]['latentVarianceRatio']:.4e} | "
                f"minJacRank={results[l]['minJacRank']} | "
                f"orderingMinRho={results[l]['orderingMinRho']:.3f}"
            )

        return results
    def largestEncodingErrorPercent(self):
        self.vaeNet.encoder.isTraining = False
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            diffMatrix = self.scaledMaterialData - decoded
            largestErrorPercent = 100.0 * torch.max(torch.abs(diffMatrix)).item()
        self.vaeNet.encoder.isTraining = True
        return largestErrorPercent

    def printYieldStrengthDropAtTempLimit(self):
        if 'Y0' not in self.materialAttributes or 'Temp_Limit' not in self.materialAttributes:
            print("ERROR: Missing required attributes (Y0 or Temp_Limit)")
            return

        print("\n" + "=" * 60)
        print("Yield Strength Drop at Temperature Limit")
        print("=" * 60)
        print("{:<20} {:>12} {:>12} {:>12} {:>12}".format("Material", "T_limit (C)", "Y0 (Pa)", "Y@T_limit", "Y0/Y ratio"))
        print("-" * 60)

        for i, name in enumerate(self.materialNames):
            Y0 = self.rawData[i, self.materialAttributes['Y0']['idx']]
            Y1 = self.rawData[i, self.materialAttributes['Y1']['idx']]
            Y2 = self.rawData[i, self.materialAttributes['Y2']['idx']]
            Y3 = self.rawData[i, self.materialAttributes['Y3']['idx']]
            Temp_Limit = self.rawData[i, self.materialAttributes['Temp_Limit']['idx']]

            Y_at_limit = logBezierInterpolation(Temp_Limit, Y0, Y1, Y2, Y3)
            ratio = Y0 / Y_at_limit

            print("{:<20} {:>12.1f} {:>12.3e} {:>12.3e} {:>12.2f}".format(name, Temp_Limit, Y0, Y_at_limit, ratio))
        print("=" * 60 + "\n")

    def printModulusDropAtTempLimit(self):
        if 'E0' not in self.materialAttributes or 'Temp_Limit' not in self.materialAttributes:
            print("ERROR: Missing required attributes (E0 or Temp_Limit)")
            return

        print("\n" + "=" * 60)
        print("Young's Modulus Drop at Temperature Limit")
        print("=" * 60)
        print("{:<20} {:>12} {:>12} {:>12} {:>12}".format("Material", "T_limit (C)", "E0 (Pa)", "E@T_limit", "E0/E ratio"))
        print("-" * 60)

        for i, name in enumerate(self.materialNames):
            E0 = self.rawData[i, self.materialAttributes['E0']['idx']]
            E1 = self.rawData[i, self.materialAttributes['E1']['idx']]
            E2 = self.rawData[i, self.materialAttributes['E2']['idx']]
            E3 = self.rawData[i, self.materialAttributes['E3']['idx']]
            Temp_Limit = self.rawData[i, self.materialAttributes['Temp_Limit']['idx']]

            E_at_limit = logBezierInterpolation(Temp_Limit, E0, E1, E2, E3)
            ratio = E0 / E_at_limit

            print("{:<20} {:>12.1f} {:>12.3e} {:>12.3e} {:>12.2f}".format(name, Temp_Limit, E0, E_at_limit, ratio))
        print("=" * 60 + "\n")

    def printEncodingErrors(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            diffMatrix = self.scaledMaterialData - decoded

        attribute_names = list(self.materialAttributes.keys())
        print("-" * 40)
        print("{:<25} {:>15}".format("Attribute", "Max % Error"))
        print("-" * 40)
        col = 0
        for name in attribute_names:
            percent_err = 100 * diffMatrix[:, col].abs().numpy()
            max_percent_err = np.max(percent_err)
            print("{:<25} {:>15.6f}".format(name, max_percent_err))
            col += 1
        print("-" * 40)
    def compute_parameter_norm(self):
        total_sq = 0.0
        with torch.no_grad():
            for p in self.vaeNet.parameters():
                total_sq += torch.sum(p.detach() ** 2).item()
        return total_sq ** 0.5
    def loadAutoencoderFromFile(self, fileName):
        self.vaeNet.load_state_dict(torch.load(fileName))
        self.vaeNet.eval()
        self.training_latents = self.vaeNet.encoder(self.scaledMaterialData).cpu()
    def _compute_vae_loss(self, klFactor):
        predData = self.vaeNet(self.scaledMaterialData)
        klLoss = klFactor * self.vaeNet.encoder.kl
        reconLoss = ((self.scaledMaterialData - predData) ** 2).sum()
        loss = reconLoss + klLoss
        return reconLoss, klLoss, loss
    def trainAutoencoder(
        self,
        numEpochs,
        klFactor,
        savedNet,
        learningRate,
        maxAttributeErrorPercent=0.0005,
        sharpAwareMinimization=False,
        sam_rho=0.05,
        Sam_diagnostics=False,
    ):
        opt = torch.optim.AdamW(self.vaeNet.parameters(), learningRate)
        convgHistory = {'reconLoss': [], 'klLoss': [], 'loss': []}
        self.vaeNet.encoder.isTraining = True

        if sharpAwareMinimization and Sam_diagnostics:
            theta_norm0 = self.compute_parameter_norm()
            print(f"[SAM DIAG] Initial ||theta||_2 = {theta_norm0:.6e}")
            print(f"[SAM DIAG] Initial rho/||theta||_2 = {sam_rho / max(theta_norm0, 1e-12):.6e}")

        for epoch in range(numEpochs):
            if not sharpAwareMinimization:
                opt.zero_grad()
                reconLoss, klLoss, loss = self._compute_vae_loss(klFactor)
                loss.backward()
                opt.step()

            else:
                # ---- first pass: compute gradient at current weights ----
                opt.zero_grad()
                reconLoss, klLoss, loss = self._compute_vae_loss(klFactor)
                loss.backward()

                grad_norm_sq = 0.0
                for p in self.vaeNet.parameters():
                    if p.grad is not None:
                        grad_norm_sq += torch.sum(p.grad.detach() ** 2).item()
                grad_norm = (grad_norm_sq ** 0.5) + 1e-12

                # save perturbations and move to adversarial weights
                e_ws = []
                with torch.no_grad():
                    for p in self.vaeNet.parameters():
                        if p.grad is None:
                            e_ws.append(None)
                            continue
                        e_w = sam_rho * p.grad / grad_norm
                        p.add_(e_w)
                        e_ws.append(e_w)

                # ---- second pass: gradient at perturbed weights ----
                opt.zero_grad()
                reconLoss, klLoss, loss = self._compute_vae_loss(klFactor)
                loss.backward()

                # restore original weights
                with torch.no_grad():
                    for p, e_w in zip(self.vaeNet.parameters(), e_ws):
                        if e_w is not None:
                            p.sub_(e_w)

                # optimizer step using gradient from perturbed weights
                opt.step()

            convgHistory['reconLoss'].append(reconLoss.item())
            convgHistory['klLoss'].append((klLoss.item() / klFactor) if klFactor != 0 else 0.0)
            convgHistory['loss'].append(loss.item())

            largestErrorPercent = self.largestEncodingErrorPercent()
            if largestErrorPercent < maxAttributeErrorPercent:
                if sharpAwareMinimization and Sam_diagnostics:
                    theta_norm = self.compute_parameter_norm()
                    print(
                        'Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, '
                        'maxPercentErr {:.3e}, ||theta|| {:.3e}, rho/||theta|| {:.3e}'.format(
                            epoch, reconLoss.item(), klLoss.item(), loss.item(),
                            largestErrorPercent, theta_norm, sam_rho / max(theta_norm, 1e-12)
                        )
                    )
                else:
                    print('Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, maxPercentErr {:.3e}'.format(
                        epoch, reconLoss.item(), klLoss.item(), loss.item(), largestErrorPercent))
                print("Converged!")
                break

            if (epoch % 5000 == 0):
                tag = "SAM" if sharpAwareMinimization else "STD"
                if sharpAwareMinimization and Sam_diagnostics:
                    theta_norm = self.compute_parameter_norm()
                    print(
                        '[{}] Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, '
                        'maxPercentErr {:.3e}, ||theta|| {:.3e}, rho/||theta|| {:.3e}'.format(
                            tag, epoch, reconLoss.item(), klLoss.item(), loss.item(),
                            largestErrorPercent, theta_norm, sam_rho / max(theta_norm, 1e-12)
                        )
                    )
                else:
                    print('[{}] Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, maxPercentErr {:.3e}'.format(
                        tag, epoch, reconLoss.item(), klLoss.item(), loss.item(), largestErrorPercent))

        self.vaeNet.encoder.isTraining = False
        torch.save(self.vaeNet.state_dict(), savedNet)
        self.training_latents = self.vaeNet.encoder(self.scaledMaterialData).cpu()
        return convgHistory

    def getMaterialName(self, index):
        if 0 <= index < len(self.materialNames):
            return self.materialNames[index]
        else:
            raise IndexError("Index out of range for material names.")

    def getValuesAtLatentPoints(self, attributeName, zPts):
        decoded = self.vaeNet.decoder(zPts)
        material_properties = self.getMaterialProperties(decoded)
        return material_properties[attributeName].detach().numpy()

    def getMaterialPropertyAtTemperature(self, name, zPts, T):
        decoded = self.vaeNet.decoder(zPts)
        material_properties = self.getMaterialProperties(decoded)
        if name in ['E', 'Y']:
            M0 = material_properties[name + '0'].detach().numpy()
            M1 = material_properties[name + '1'].detach().numpy()
            M2 = material_properties[name + '2'].detach().numpy()
            M3 = material_properties[name + '3'].detach().numpy()
            M = logBezierInterpolation(T, M0, M1, M2, M3)
        elif name == 'K':
            M0 = material_properties[name + '0'].detach().numpy()
            M1 = material_properties[name + '1'].detach().numpy()
            M2 = material_properties[name + '2'].detach().numpy()
            M3 = material_properties[name + '3'].detach().numpy()
            M = bezierInterpolation(T, M0, M1, M2, M3)
        return M

    def getMaterialPropertyAtTemperatureTorch(self, name, zPts, T):
        decoded = self.vaeNet.decoder(zPts)
        material_properties = self.getMaterialProperties(decoded)
        if name in ['E', 'Y']:
            M0 = material_properties[name + '0']
            M1 = material_properties[name + '1']
            M2 = material_properties[name + '2']
            M3 = material_properties[name + '3']
            M = logBezierInterpolation_torch(T, M0, M1, M2, M3)
        elif name == 'K':
            M0 = material_properties[name + '0']
            M1 = material_properties[name + '1']
            M2 = material_properties[name + '2']
            M3 = material_properties[name + '3']
            M = bezierInterpolation_torch(T, M0, M1, M2, M3)
        return M

    def getValueOfAttributeAtZLocationAtTemperature(self, attributeName, zPts, T, compute_gradients=False):
        if compute_gradients:
            zPts_tensor = torch.tensor(zPts, dtype=torch.float32, requires_grad=True) if not isinstance(zPts, torch.Tensor) else zPts.clone().detach().requires_grad_(True)
            decoded = self.vaeNet.decoder(zPts_tensor)
            material_properties = self.getMaterialProperties(decoded)

            if attributeName in ['E', 'Y']:
                M0 = material_properties[attributeName + '0']
                M1 = material_properties[attributeName + '1']
                M2 = material_properties[attributeName + '2']
                M3 = material_properties[attributeName + '3']
                value_at_T = logBezierInterpolation_torch(T, M0, M1, M2, M3)
            elif attributeName == 'K':
                M0 = material_properties[attributeName + '0']
                M1 = material_properties[attributeName + '1']
                M2 = material_properties[attributeName + '2']
                M3 = material_properties[attributeName + '3']
                value_at_T = bezierInterpolation_torch(T, M0, M1, M2, M3)
            else:
                value_at_T = material_properties[attributeName]

            value_at_T.sum().backward()
            gradient = zPts_tensor.grad.clone()

            return value_at_T.detach().numpy(), gradient.detach().numpy().T
        else:
            with torch.no_grad():
                decoded = self.vaeNet.decoder(zPts)
                material_properties = self.getMaterialProperties(decoded)

                if attributeName in ['E', 'Y']:
                    M0 = material_properties[attributeName + '0']
                    M1 = material_properties[attributeName + '1']
                    M2 = material_properties[attributeName + '2']
                    M3 = material_properties[attributeName + '3']
                    value_at_T = logBezierInterpolation_torch(T, M0, M1, M2, M3)
                elif attributeName == 'K':
                    M0 = material_properties[attributeName + '0']
                    M1 = material_properties[attributeName + '1']
                    M2 = material_properties[attributeName + '2']
                    M3 = material_properties[attributeName + '3']
                    value_at_T = bezierInterpolation_torch(T, M0, M1, M2, M3)
                else:
                    value_at_T = material_properties[attributeName]

            return value_at_T.detach().numpy()

    def getMaterialPropertiesAtLatentPoints(self, zPts, compute_gradients=False):
        if compute_gradients:
            zPts_tensor = torch.tensor(zPts, dtype=torch.float32) \
                        if not isinstance(zPts, torch.Tensor) \
                        else zPts.clone().detach()

            def compute_single_element(z_elem):
                z_batch = z_elem.unsqueeze(0)
                decoded = self.vaeNet.decoder(z_batch)
                props = self.getMaterialProperties(decoded)
                return torch.stack([props[name][0] for name in sorted(props.keys())])

            jac_fn = vmap(jacrev(compute_single_element))
            jacobians = jac_fn(zPts_tensor)

            with torch.no_grad():
                decoded = self.vaeNet.decoder(zPts_tensor)
                material_properties = self.getMaterialProperties(decoded)

            gradients = {}
            for idx, prop_name in enumerate(sorted(material_properties.keys())):
                gradients[prop_name] = jacobians[:, idx, :]

            return material_properties, gradients
        else:
            with torch.no_grad():
                decoded = self.vaeNet.decoder(zPts)
                material_properties = self.getMaterialProperties(decoded)
            return material_properties

    def getMaterialProperties(self, decoded):
        properties = {}
        for name, attribute in self.materialAttributes.items():
            idx = attribute['idx']
            scaleMax = attribute['scaleMax']
            scaleMin = attribute['scaleMin']
            if name in ['E0', 'E1', 'E2', 'E3', 'Y0', 'Y1', 'Y2', 'Y3']:
                properties[name] = 10 ** (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
            else:
                properties[name] = (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
        return properties

    def getClosestRealMaterialIndex(self, zDesign):
        with torch.no_grad():
            zReal = self.vaeNet.encoder(self.scaledMaterialData)

        distances = torch.cdist(zDesign, zReal)
        closest_indices = torch.argmin(distances, dim=1)
        return closest_indices

    def plotMaterialHistogram(self, closest_indices, xDesign, colors):
        plt.figure(figsize=(4, 4))

        xDesign_np = np.array(xDesign)
        mask = xDesign_np >= 0.1
        filtered_indices = closest_indices[mask]

        unique_indices, counts = np.unique(filtered_indices, return_counts=True)
        total_counts = np.sum(counts)

        all_counts = np.zeros(len(self.materialNames))
        all_counts[unique_indices] = counts

        if (colors is None) or (len(colors) < len(self.materialNames)):
            colors = cm.tab10(np.linspace(0, 1, len(self.materialNames)))

        plt.bar(range(len(self.materialNames)), 100 * all_counts / total_counts, color=colors, edgecolor='black')
        plt.xticks(range(len(self.materialNames)), self.materialNames, rotation=45, ha='right')
        plt.xlabel('Material')
        plt.ylabel('Percentage')
        plt.title('Materials Distribution')
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    def getClosestRealMaterialZValues(self, zDesign):
        with torch.no_grad():
            zReal = self.vaeNet.encoder(self.scaledMaterialData)

        distances = torch.cdist(zDesign, zReal)
        closest_indices = torch.argmin(distances, dim=1)
        return zReal[closest_indices].detach().numpy()

    def materialDistance(self, zDesign, xDesign, gamma):
        zDesign_tensor = torch.tensor(zDesign, dtype=torch.float32, requires_grad=True)
        decoded_design = self.vaeNet.decoder(zDesign_tensor)

        props_real = self.rawData
        props_design = self.getMaterialProperties(decoded_design)
        props_design = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in props_design.values()], dim=1)
        props_real = torch.tensor(props_real) if not isinstance(props_real, torch.Tensor) else props_real

        design_exp = props_design.unsqueeze(1)
        real_exp = props_real.unsqueeze(0)

        norm_diff = ((design_exp - real_exp) ** 2) / (real_exp ** 2 + 1e-12)

        net_distance = norm_diff.sum(dim=2)
        min_distances, _ = torch.min(net_distance, dim=1)

        penalty = gamma * torch.mean(min_distances * xDesign)
        penalty.backward()
        grad = zDesign_tensor.grad.detach().numpy()
        grad = grad.T.reshape(-1)
        return penalty.detach().numpy(), grad

    def materialAttributeDistance(self, attributeName, zDesign, xDesign, gamma):
        attributeId = list(self.materialAttributes.keys()).index(attributeName)
        zDesign_tensor = torch.tensor(zDesign, dtype=torch.float32, requires_grad=True)
        decoded_design = self.vaeNet.decoder(zDesign_tensor)

        props_real = self.rawData
        props_design = self.getMaterialProperties(decoded_design)
        props_design = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in props_design.values()], dim=1)
        props_real = torch.tensor(props_real) if not isinstance(props_real, torch.Tensor) else props_real

        design_exp = props_design[:, attributeId].unsqueeze(1)
        real_exp = props_real[:, attributeId].unsqueeze(0)

        norm_diff = ((design_exp - real_exp) ** 2) / (real_exp ** 2 + 1e-12)

        net_distance = norm_diff
        min_distances, _ = torch.min(net_distance, dim=1)

        penalty = gamma * torch.mean(min_distances * xDesign)
        penalty.backward()
        grad = zDesign_tensor.grad.detach().numpy()
        grad = grad.T.reshape(-1)
        return penalty.detach().numpy(), grad

    def plotLSR(self, zRealPts, zDesignPts=None, xDesign=None):
        zRealPts = np.asarray(zRealPts, dtype=float)
        if zRealPts.ndim == 1:
            zRealPts = zRealPts.reshape(-1, 1)

        if zDesignPts is not None:
            zDesignPts = np.asarray(zDesignPts, dtype=float)
            if zDesignPts.ndim == 1:
                zDesignPts = zDesignPts.reshape(-1, 1)

        latentDim = zRealPts.shape[1]

        # PCA projection only for latentDim > 3
        if latentDim > 3:
            pca = PCA(n_components=2)
            zRealPts_2d = pca.fit_transform(zRealPts)
            zDesignPts_2d = pca.transform(zDesignPts) if zDesignPts is not None else None

            plt.figure()
            plt.title(f"Latent Space (PCA projection from {latentDim}D)")
            if zDesignPts_2d is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    plt.scatter(
                        zDesignPts_2d[mask, 0],
                        zDesignPts_2d[mask, 1],
                        c='red',
                        marker='o',
                        s=20,
                        label='Optimized Materials',
                        alpha=0.2
                    )
            plt.scatter(zRealPts_2d[:, 0], zRealPts_2d[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=0.4)
            for i, label in enumerate(self.materialNames):
                plt.text(zRealPts_2d[i, 0] + 0.1, zRealPts_2d[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.grid(True)
            plt.legend()
            plt.show()

        # 3D plot
        elif latentDim == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if zDesignPts is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    ax.scatter(
                        zDesignPts[mask, 0],
                        zDesignPts[mask, 1],
                        zDesignPts[mask, 2],
                        c='red',
                        marker='o',
                        s=20,
                        label='Optimized Materials',
                        alpha=0.6
                    )
            ax.scatter(zRealPts[:, 0], zRealPts[:, 1], zRealPts[:, 2], c='black', marker='*', s=200, label='Real Materials', alpha=0.6)
            for i, label in enumerate(self.materialNames):
                ax.text(zRealPts[i, 0] + 0.1, zRealPts[i, 1], zRealPts[i, 2], str(label), fontsize=10, color='black')
            ax.set_xlabel('$z_1$')
            ax.set_ylabel('$z_2$')
            ax.set_zlabel('$z_3$')
            plt.legend()
            plt.show()

        # 2D plot
        elif latentDim == 2:
            plt.figure()
            if zDesignPts is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    plt.scatter(
                        zDesignPts[mask, 0],
                        zDesignPts[mask, 1],
                        c='red',
                        marker='o',
                        s=20,
                        label='Optimized Materials',
                        alpha=0.2
                    )
            plt.scatter(zRealPts[:, 0], zRealPts[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=0.4)
            for i, label in enumerate(self.materialNames):
                plt.text(zRealPts[i, 0] + 0.1, zRealPts[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.grid(True)
            plt.legend()
            plt.show()

        # 1D plot
        elif latentDim == 1:
            plt.figure(figsize=(8, 4))

            y_real = np.zeros(zRealPts.shape[0])
            plt.scatter(zRealPts[:, 0], y_real, c='black', marker='*', s=200, label='Real Materials', alpha=0.7)

            for i, label in enumerate(self.materialNames):
                plt.text(zRealPts[i, 0] + 0.02, y_real[i] + 0.01, str(label), fontsize=11, color='black')

            if zDesignPts is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    plt.scatter(
                        zDesignPts[mask, 0],
                        0.02 * np.ones(np.sum(mask)),
                        c='red',
                        marker='o',
                        s=20,
                        label='Optimized Materials',
                        alpha=0.2
                    )

            plt.xlabel('$z$')
            plt.yticks([])
            plt.title('1D Latent Space')
            plt.grid(True)
            plt.legend()
            plt.show()

        else:
            raise ValueError(f"Unsupported latent dimension: {latentDim}")

    def plotLSRContours(self, attributeName, title=""):
        attributeId = list(self.materialAttributes.keys()).index(attributeName)
        zReal = self.training_latents.detach().cpu().numpy() if hasattr(self.training_latents, "detach") else np.asarray(self.training_latents)
        zReal = np.asarray(zReal, dtype=float)
        if zReal.ndim == 1:
            zReal = zReal.reshape(-1, 1)

        latentDim = zReal.shape[1]
        n_points = 50

        # 1D latent space: line plot instead of contour
        if latentDim == 1:
            zmin = float(np.min(zReal[:, 0]) - 1.0)
            zmax = float(np.max(zReal[:, 0]) + 1.0)
            z1 = np.linspace(zmin, zmax, n_points)
            QOI = []

            with torch.no_grad():
                for z in z1:
                    z_tensor = torch.tensor([[z]], dtype=torch.float32)
                    decoded = self.vaeNet.decoder(z_tensor)
                    decodedValues = self.getMaterialProperties(decoded)
                    QOI.append(decodedValues[list(decodedValues.keys())[attributeId]].item())

            QOI = np.array(QOI)

            plt.figure(figsize=(7.5, 4.5))
            plt.plot(z1, QOI, linewidth=2)
            real_vals = []
            with torch.no_grad():
                for zr in zReal[:, 0]:
                    z_tensor = torch.tensor([[zr]], dtype=torch.float32)
                    decoded = self.vaeNet.decoder(z_tensor)
                    decodedValues = self.getMaterialProperties(decoded)
                    real_vals.append(decodedValues[list(decodedValues.keys())[attributeId]].item())
            real_vals = np.array(real_vals)

            plt.scatter(zReal[:, 0], real_vals, c='black', marker='*', s=200, alpha=1.0)
            for i, label in enumerate(self.materialNames):
                plt.text(zReal[i, 0] + 0.03, real_vals[i], str(label), fontsize=10, color='black')

            plt.xlabel('$z$')
            plt.ylabel(list(self.materialAttributes.keys())[attributeId])
            plt.title(title if title else f"{attributeName} over 1D latent space")
            plt.grid(True)
            plt.show()
            return

        if latentDim > 2:
            pca = PCA(n_components=2)
            zReal_2d = pca.fit_transform(zReal)
            z1 = np.linspace(-5, 5, n_points)
            z2 = np.linspace(-5, 5, n_points)
            Z1, Z2 = np.meshgrid(z1, z2)
            Z_grid_2d = np.stack([Z1.ravel(), Z2.ravel()], axis=1)
            Z_grid_full = pca.inverse_transform(Z_grid_2d)
        else:
            zReal_2d = zReal
            z1 = np.linspace(-5, 5, n_points)
            z2 = np.linspace(-5, 5, n_points)
            Z1, Z2 = np.meshgrid(z1, z2)
            Z_grid_full = np.stack([Z1.ravel(), Z2.ravel()], axis=1)

        QOI = []
        with torch.no_grad():
            for z in Z_grid_full:
                z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
                decoded = self.vaeNet.decoder(z_tensor)
                decodedValues = self.getMaterialProperties(decoded)
                QOI.append(decodedValues[list(decodedValues.keys())[attributeId]].item())
        QOI = np.array(QOI).reshape(Z1.shape)

        plt.figure(figsize=(7.5, 6))
        contour = plt.contourf(Z1, Z2, QOI, levels=30, cmap='viridis')
        units = self.materialAttributes[list(self.materialAttributes.keys())[attributeId]]['unit']
        plt.colorbar(contour, label=list(self.materialAttributes.keys())[attributeId] + " (" + units + ")")
        plt.scatter(zReal_2d[:, 0], zReal_2d[:, 1], c='black', marker='*', s=200, alpha=1.0)
        for i, label in enumerate(self.materialNames):
            plt.text(zReal_2d[i, 0] + 0.1, zReal_2d[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        plt.title(title)
        plt.show()

    def plotTemperatureVsMaterialProperty(self, attrName, semilogy=False):
        zRealPts = self.vaeNet.encoder.z
        plt.figure()
        T = np.linspace(TMin, TMax, 300)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
        for i in range(zRealPts.shape[0]):
            zPt = zRealPts[i, :].view(1, -1)
            M = self.getMaterialPropertyAtTemperature(attrName, zPt, T)
            marker = markers[i % len(markers)]
            if semilogy:
                plt.semilogy(T, M, label=self.materialNames[i], marker=marker, markevery=30)
            else:
                plt.plot(T, M, label=self.materialNames[i], marker=marker, markevery=30)

        plt.xlabel("Temperature (C)")
        plt.ylabel(f"{attrName}")
        plt.title(f"Temperature vs {attrName}")
        plt.legend(self.materialNames)
        plt.grid()
        plt.show()

    def plotTemperatureVsMaterialPropertyRaw(self, attrName, semilogy=False, colors=None):
        plt.figure()
        T = np.linspace(TMin, TMax, 300)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']

        if (colors is None) or (len(colors) < self.rawData.shape[0]):
            colors = ['#0201fc', '#3cb44b', '#654321']
            if self.rawData.shape[0] > 3:
                additional_colors = cm.tab10(np.linspace(0.3, 1, self.rawData.shape[0] - 3))
                colors.extend([cm.colors.rgb2hex(c) for c in additional_colors])

        for i in range(self.rawData.shape[0]):
            if attrName in ['E', 'Y']:
                M0 = self.rawData[i, self.materialAttributes[attrName + '0']['idx']]
                M1 = self.rawData[i, self.materialAttributes[attrName + '1']['idx']]
                M2 = self.rawData[i, self.materialAttributes[attrName + '2']['idx']]
                M3 = self.rawData[i, self.materialAttributes[attrName + '3']['idx']]
                M = logBezierInterpolation(T, M0, M1, M2, M3)
            elif attrName == 'K':
                M0 = self.rawData[i, self.materialAttributes[attrName + '0']['idx']]
                M1 = self.rawData[i, self.materialAttributes[attrName + '1']['idx']]
                M2 = self.rawData[i, self.materialAttributes[attrName + '2']['idx']]
                M3 = self.rawData[i, self.materialAttributes[attrName + '3']['idx']]
                M = bezierInterpolation(T, M0, M1, M2, M3)
            else:
                continue

            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            if semilogy:
                plt.semilogy(T, M, label=self.materialNames[i], marker=marker, markevery=30, color=color)
            else:
                plt.plot(T, M, label=self.materialNames[i], marker=marker, markevery=30, color=color)

        plt.xlabel("Temperature (C)")
        plt.ylabel(f"{attrName}")
        plt.title(f"Temperature vs {attrName} (Raw Data)")
        plt.legend(self.materialNames)
        plt.grid()
        plt.show()

    def getHeaviestMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        heaviest_idx = np.argmax(density_values)
        heaviest_z = z_real[heaviest_idx].detach().cpu().numpy()
        return heaviest_z

    def getLightestMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        lightest_idx = np.argmin(density_values)
        lightest_z = z_real[lightest_idx].detach().cpu().numpy()
        return lightest_z

    def getStrongestMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        yield_values = decoded_properties['Y0'].detach().cpu().numpy().flatten()
        strongest_idx = np.argmax(yield_values)
        strongest_z = z_real[strongest_idx].detach().cpu().numpy()
        return strongest_z

    def getBestStiffnessToDensityMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        stiffness_values = decoded_properties['E0'].detach().cpu().numpy().flatten()
        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        best_idx = np.argmax(stiffness_values / density_values)
        best_z = z_real[best_idx].detach().cpu().numpy()
        return best_z

    def getBestStrengthToDensityMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        yield_values = decoded_properties['Y0'].detach().cpu().numpy().flatten()
        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        best_idx = np.argmax(yield_values / density_values)
        best_z = z_real[best_idx].detach().cpu().numpy()
        return best_z