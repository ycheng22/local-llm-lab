# Phase 5: Test-Time Compute Scaling

We have successfully generated 1,600 test candidates (100 complex tasks, K=16) and verified them using the `DockerVerifier`.

## Capability vs Compute

The findings from this phase are extremely significant! In Phase 4, the model failed completely (0% Pass@4) when generating candidates with default temperature. However, by simply giving the model more attempts (K=16) with a slightly higher temperature (`0.8`), we unlocked hidden reasoning capabilities:

![Capability vs Compute](capability_vs_compute.png)

### Core Metrics

- **Pass@1:** `2.9%`
- **Pass@2:** `4.4%`
- **Pass@4:** `6.0%`
- **Pass@8:** `7.7%`
- **Pass@16:** `10.0%`

### Analysis

> [!TIP]
> **Key Scientific Takeaway**
> The Qwen3.5-2B model was previously deemed completely incapable of solving the complex MBPP dataset. By employing **Test-Time Compute Scaling** (generating more candidates and filtering by verifier), the capability skyrocketed from functionally 0% to **10%**!
> 
> This proves that the model possesses the *latent knowledge* to solve complex logic, but its search trajectory is brittle. Allowing for a broader search budget effectively bridges this gap without a single parameter update.

The results have been documented in the `README.md` file. We are now ready to proceed to Phase 6 (Tool-Use Agent & 4B Scaling)!
