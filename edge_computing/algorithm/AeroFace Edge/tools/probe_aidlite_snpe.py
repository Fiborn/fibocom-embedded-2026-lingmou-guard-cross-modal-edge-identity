import os, aidlite

MODEL_PATH = os.path.abspath("../model/scrfd_2.5g_kps.dlc")

model = aidlite.Model.create_instance(MODEL_PATH)
assert model is not None

for impl_name in ["TYPE_LOCAL", "TYPE_DEFAULT"]:
    for acc_name in ["TYPE_DSP", "TYPE_GPU", "TYPE_CPU"]:
        if not (hasattr(aidlite.ImplementType, impl_name) and hasattr(aidlite.AccelerateType, acc_name)):
            continue
        cfg = aidlite.Config.create_instance()
        cfg.framework_type = aidlite.FrameworkType.TYPE_QNN
        cfg.implement_type = getattr(aidlite.ImplementType, impl_name)
        cfg.accelerate_type = getattr(aidlite.AccelerateType, acc_name)
        cfg.number_of_threads = 4
        cfg.is_quantify_model = 0
        cfg.fast_timeout = -1

        itp = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, cfg)
        print("try", impl_name, acc_name, "build", itp is not None)
        if itp is None:
            continue
        r = itp.init()
        print(" init=", r)
        if r != 0:
            continue
        r = itp.load_model()
        print(" load_model=", r)
        if r == 0:
            print("✅ QNN works:", impl_name, acc_name)
            raise SystemExit(0)

print("❌ QNN not working with this dlc")