# V0.1 可复现运行环境

这里保存的是依赖来源与哈希锁，不保存数 GB 的 wheel。`artifacts/*.json` 中的
`metadata_resolved_not_installed` 是锁生成时的状态，**不代表当前机器已经安装或通过验收**；
实际下载结果由 fetch 输出证明，安装结果由 finalized toolchain manifest 与运行时 probe 证明。

在线下载和离线安装是两个显式步骤；普通模型加载、bootstrap 和 runner 都不会自动下载依赖或权重。
下载只使用公开的官方来源，不读取 Hugging Face token，也不修改全局 Python、驱动或代理设置。

## CPU 学习环境

Windows 的锁定解释器为 CPython 3.12.14，Ubuntu 的为 CPython 3.12.3。
`v01-cpu` 根据当前操作系统选择对应 profile；版本不符会明确拒绝，不会静默换版本。

GitHub-hosted Windows development CI uses a separate explicit
`py312-win-amd64-cpu-ci` profile with official CPython 3.12.10. The
[Actions Python manifest](https://github.com/actions/python-versions/blob/main/versions-manifest.json)
does not provide a Windows 3.12.14 binary. This CI-only profile
keeps the same 25 runtime wheel bytes, versions and dependency closure, but has
its own target, marker environment and lock digest. It does not change the
`v01-cpu` alias or the exact 3.12.14 Windows qualification environment.

The CI-only profile runs independent development tests and portable release
evidence readers; it is not in the three-profile release runtime whitelist and
cannot seal or grant a V0.1 release qualification. Its package compatibility is
not proof that 3.12.10 executed the formal 3.12.14 qualification. Actual CI
installation/import/test results must be checked separately from the lock.

```sh
python scripts/fetch_runtime.py --runtime-profile v01-cpu
python scripts/bootstrap.py --profile v01-dev --runtime-profile v01-cpu --locked --offline --write-manifest out/v0.1/toolchain.json
python scripts/in_env.py --manifest out/v0.1/toolchain.json -- illm model inspect --model fixture:tiny-decoder
```

V0.0 的默认最小环境保持不变：不传 `--runtime-profile` 仍只安装仓库内的基础开发 wheel。
安装目录只在 `.illm/envs/<profile>/` 内；重新 bootstrap 同名 profile 会重建这一托管环境。
缓存缺失、损坏或目标平台错误会在重建前拒绝。
离线安装关闭预编译全部第三方 `.pyc`（`--no-compile`），减少 WSL `/mnt/c` 上的文件系统开销；
需要的模块仍会由解释器正常导入。环境安装耗时不属于模型推理性能基线。

## WSL CUDA 环境

```sh
python3 scripts/fetch_runtime.py --runtime-profile v01-cuda
python3 scripts/bootstrap.py --profile v01-gpu-dev --runtime-profile v01-cuda --locked --offline --write-manifest out/v0.1/gpu-dev-toolchain.json
```

CUDA profile 锁定 Torch 2.8.0+cu126，而非依据机器驱动临时解析依赖。
当前 device probe 明确限定首个验收设备 SM 8.6（RTX 3070）；它不是所有 CUDA GPU 的兼容性声明。
其他 GPU 需要新增或扩展显式设备验收规则，不能把未验证的设备视为已支持。

## 固定真实模型

```sh
python scripts/fetch_model.py --model-profile configs/models/v0_1.yaml
```

模型固定到不可变 revision。每个文件验证字节数与 LFS SHA-256 或 Git blob SHA-1，
并在成功后额外记录实际 SHA-256 的 `FETCH-COMPLETE.json`。已有缓存先重新验签，
损坏时拒绝覆盖；全部命中时零网络请求。失败的临时文件不会变成可加载的最终文件。
默认下载预算约 1.1 GB；依赖 fetch 默认预算约 4.5 GB。缓存位于 `.illm/`，不提交 Git。

源代码、锁、依赖包或设备身份改变后，runner 会拒绝旧 manifest。开发中的直接测试可以使用
托管环境解释器和 `PYTHONPATH=src`，但这类运行不等于源码冻结后的正式发布验收。
