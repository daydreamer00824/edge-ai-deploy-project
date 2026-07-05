import argparse
import yaml
from pathlib import Path
import logging
from timer import Timerecorder, time_block
from export_model import export_model
from inspect_onnx import inspect_onnx
from image_process import collect_image
import onnxruntime as ort
import numpy as np
from check_onnx import check_onnx
from postprocess import get_topk
from benchmark import save_benchmark, save_benchmark_summary

def use_argparse():
    parse = argparse.ArgumentParser(description="程序自动测试batch_size")
    parse.add_argument("--input", type=str, required=True, help="图片输入目录")
    parse.add_argument("--output", type=str, required=True, help="输出目录")
    parse.add_argument("--models", type=str, required=True, help="模型导出目录")
    parse.add_argument("--config", type=str, default="/home/daydreamer/Desktop/study/python-onnx-demo/config", help="config目录")
    parse.add_argument("--log", type=str, default="/home/daydreamer/Desktop/study/python-onnx-demo/logs", help="日志目录")
    parse.add_argument("--batch_size", type=int, nargs="+", default=[1, 4, 8, 12, 16, 32], 
                       help="要测试的 batch_size 列表，例如: --batch-sizes 1 4 8 12 16 32")
    parse.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="每个 batch_size 重复测试次数"
    )
    parse.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="每个 batch_size 重正式测试前的预热次数"
    )
    parse.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        help="ONNX Runtime Execution Provider，例如 CPUExecutionProvider 或 CUDAExecutionProvider"
    )
    return parse.parse_args()

def load_config(config_dir:Path):
    if not config_dir.exists():
        raise FileNotFoundError(f"配置目录不存在: {config_dir}")
    
    config_name = Path("config.yaml")
    config_path = config_dir / config_name

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if cfg is None:
                raise ValueError(f"配置文件为空: {config_path}")
            else:
                return cfg
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在{config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件错误{e}")
    

def set_logging(log_dir : Path):
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_name = Path("run_test.log")
    log_path = log_dir / log_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()]
    )

def run_one_benchmark(session, input_name, output_name, image, batch_size: int, provider):
    timer = Timerecorder()


    for start in range(0, len(image), batch_size):
        end = start + batch_size
        image_batch = image[start:end]
        input_batch = np.stack(image_batch, axis=0).astype(np.float32)

        with time_block(timer, "infer"):
            result = session.run([output_name], {input_name: input_batch})[0]

        with time_block(timer, "postprocess"):
            topk_indices, topk_scores = get_topk(result, k=5)

    image_count = len(image)

    infer_time = timer.get("infer")
    postprocess_time = timer.get("postprocess")

    avg_infer_ms = infer_time / image_count * 1000
    throughput = image_count / infer_time

    benchmark_result = {
        "provider":provider,
        "batch_size": batch_size,
        "image_count": image_count,
        "preprocess_time": 0.0,
        "infer_time": round(infer_time, 6),
        "postprocess_time": round(postprocess_time, 6),
        "save_result_time": 0.0,
        "avg_infer_ms": round(avg_infer_ms, 3),
        "throughput": round(throughput, 3),
    }

    return benchmark_result

def run_warmup(session, input_name, output_name, image, batch_size, warmup):
    if warmup <= 0:
        return
    
    for warmup_id in range(1, warmup +1 ):
        for start in range(0, len(image), batch_size):
            end = start + batch_size
            image_batch = image[start : end]
            input_batch = np.stack(image_batch, axis=0).astype(np.float32)

            session.run([output_name], {input_name:input_batch})
        
        logging.info(f"warmup 完成: batch_size={batch_size}, warmup_id={warmup_id}/{warmup}")

def check_provider(provider):
    available_providers = ort.get_available_providers()
    logging.info(f"当前环境可用 Provider: {available_providers}")

    if provider not in available_providers:
        raise RuntimeError(f"指定的 provider 不可用: {provider}, 当前可用: {available_providers}")
            
        

def main():
    arg = use_argparse()
    set_logging(Path(arg.log))

    try:
        cfg = load_config(Path(arg.config))
        logging.info(f"配置文件加载成功:{cfg}")
    except Exception as e:
        logging.error(f"配置文件加载失败:{e}")
        return

    model_dir = Path(cfg["base_path"]) / Path(arg.models)
    model_path = model_dir / "model.onnx"
    try:
        if not model_path.exists():                                ####!!!!
            logging.info("未检测到 ONNX 模型，开始导出")
            model_path = export_model(model_dir)
            logging.info("模型导出成功")
        else:
            logging.info(f"检测到已有模型，直接使用: {model_path}")    ####!!!!
    except Exception as e:
        logging.error(f"模型导出失败: {e}")
        return
    
    try:
        check_onnx(model_path)
    except Exception as e:
        logging.error(f"onnx文件出错:{e}")
        return
    
    try:    
        inspect_onnx(model_path)
    except Exception as e:
        logging.error(f"ONNX 检查失败: {e}")
        return
    
    extensions = set(ext.lower() for ext in cfg.get("extensions", []))
    size = tuple(cfg.get("size", [224, 224]))

    input_path = Path(cfg["base_path"]) / Path(arg.input)
    output_path = Path(cfg["base_path"]) / Path(arg.output)

    preprocess_timer = Timerecorder()

    try:
        with time_block(preprocess_timer, "preprocess"):
            image, image_path = collect_image(input_path, output_path, extensions, size)#!
            if not image:
                logging.error("没有收集到可用于benchmark的图片")
                return#!
            
        preprocess_time = preprocess_timer.get("preprocess")
        logging.info(f"图片预处理完成，共 {len(image)} 张")
        logging.info(f"预处理总耗时: {preprocess_time:.6f} s")
    except Exception as e:
        logging.error(f"图片处理失败: {e}")
        return#!
    
    try:
        check_provider(arg.provider)
        session = ort.InferenceSession(str(model_path), providers=[arg.provider])
        actual_provider = session.get_providers()
        logging.info(f"Session 实际使用 Provider: {actual_provider}")

        output_name = session.get_outputs()[0].name
        input_name = session.get_inputs()[0].name
    except Exception as e:
        logging.error(f"ONNX Runtime Session 创建失败: {e}")
        return
    
    logging.info("*" * 50)
    logging.info(f"开始 batch_size benchmark，测试列表: {arg.batch_size}")
    logging.info("*" * 50)

    summary_resluts = []

    benchmark_path = output_path / "benchmark_results.csv"
    benchmark_summary_path = output_path / "benchmark_summary.csv"

    for old_file in [benchmark_path, benchmark_summary_path]:
        if old_file.exists():
            old_file.unlink()
            logging.info(f"已删除旧 benchmark 文件: {old_file}")

    for batch_size in arg.batch_size:
        avg_infer_ms_list = []
        throughput_list = []

        try:
            logging.info(f"开始 warmup: batch_size={batch_size}, warmup={arg.warmup}")

            run_warmup(
                session=session,
                input_name=input_name,
                output_name=output_name,
                image=image,
                batch_size=batch_size,
                warmup=arg.warmup
            )

        except Exception as e:
            logging.error(f"batch_size={batch_size} warmup 失败: {e}")
            continue

        for repeat_id in range(1, arg.repeat + 1):
            try:
                logging.info(f"开始测试 batch_size={batch_size}, repeat={repeat_id}/{arg.repeat}")

                benchmark_result = run_one_benchmark(
                    session=session,
                    input_name=input_name,
                    output_name=output_name,
                    image=image,
                    batch_size=batch_size,
                    provider=arg.provider
                )

                benchmark_result["repeat_id"] = repeat_id
                benchmark_result["preprocess_time"] = round(preprocess_time, 6)

                avg_infer_ms_list.append(benchmark_result["avg_infer_ms"])
                throughput_list.append(benchmark_result["throughput"])

                save_timer = Timerecorder()
                with time_block(save_timer, "save_benchmark"):
                    benchmark_path = save_benchmark(benchmark_result, output_path)

                logging.info(f"batch_size={batch_size}, repeat={repeat_id} 测试完成")
                logging.info(f"图片总数: {benchmark_result['image_count']}")
                logging.info(f"ONNX 推理总耗时: {benchmark_result['infer_time']} s")
                logging.info(f"后处理总耗时: {benchmark_result['postprocess_time']} s")
                logging.info(f"单图平均 ONNX 推理耗时: {benchmark_result['avg_infer_ms']} ms/image")
                logging.info(f"吞吐量: {benchmark_result['throughput']} image/s")
                logging.info(f"benchmark 文件路径: {benchmark_path}")
                logging.info("-" * 50)

            except Exception as e:
                logging.error(f"batch_size={batch_size}, repeat={repeat_id} benchmark 失败: {e}")

        if avg_infer_ms_list and throughput_list:
            avg_infer_ms_mean = float(np.mean(avg_infer_ms_list))
            throughput_mean = float(np.mean(throughput_list))

            avg_infer_ms_std = float(np.std(avg_infer_ms_list))
            throughput_std = float(np.std(throughput_list))

            summary_reslut = {
                "provider":arg.provider,
                "batch_size":batch_size,
                "repeat_count":len(avg_infer_ms_list),
                "image_count":len(image),
                "avg_infer_ms_mean":round(avg_infer_ms_mean, 3),
                "avg_infer_ms_std":round(avg_infer_ms_std, 3),
                "throughput_mean":round(throughput_mean, 3),
                "throughput_std":round(throughput_std, 3),
                "best_by_throughput":False
            }

            summary_resluts.append(summary_reslut)

            logging.info("=" * 50)
            logging.info(f"batch_size={batch_size} 重复测试完成")
            logging.info(f"有效测试次数: {len(avg_infer_ms_list)} / {arg.repeat}")
            logging.info(f"平均单图 ONNX 推理耗时: {avg_infer_ms_mean:.3f} ms/image")
            logging.info(f"单图推理耗时标准差: {avg_infer_ms_std:.3f} ms/image")
            logging.info(f"平均吞吐量: {throughput_mean:.3f} image/s")
            logging.info(f"吞吐量标准差: {throughput_std:.3f} image/s")
            logging.info("=" * 50)
    
    logging.info("全部 batch_size benchmark 完成")
    if summary_resluts:
        best_result = max(summary_resluts, key=lambda x:x["throughput_mean"])

        for item in summary_resluts:
            item["best_by_throughput"] = item["batch_size"] == best_result["batch_size"]

        summary_path = save_benchmark_summary(summary_resluts, output_path)

        logging.info("*" * 50)
        logging.info("benchmark 汇总完成")
        logging.info(f"最优 batch_size: {best_result['batch_size']}")
        logging.info(f"最优平均吞吐量: {best_result['throughput_mean']} image/s")
        logging.info(f"对应平均单图推理耗时: {best_result['avg_infer_ms_mean']} ms/image")
        logging.info(f"benchmark 汇总文件路径: {summary_path}")
        logging.info("*" * 50)

    else:
        logging.error("没有有效 benchmark 结果，无法生成汇总文件")



if __name__ == "__main__":
    main()