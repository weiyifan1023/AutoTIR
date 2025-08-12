from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse


def zero_shot(args, config_dict):
    # naive generation
    from flashrag.pipeline import SequentialPipeline
    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate

    templete = PromptTemplate(
        config=config,
        system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
        user_prompt="Question: {question}",
    )
    pipeline = SequentialPipeline(config, templete)
    result = pipeline.naive_run(test_data)


def naive(args, config_dict):
    # naive rag
    from flashrag.pipeline import SequentialPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    pipeline = SequentialPipeline(config)

    result = pipeline.medrun(test_data)


def iterretgen(args, config_dict):
    """
    Reference:
        Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative
                            Retrieval-Generation Synergy"
        in EMNLP Findings 2023.

        Zhangyin Feng et al. "Retrieval-Generation Synergy Augmented Large Language Models"
        in EMNLP Findings 2023.
    """
    iter_num = 3

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import IterativePipeline

    pipeline = IterativePipeline(config, iter_num=iter_num)
    result = pipeline.run(test_data)


def ircot(args, config_dict):
    """
    Reference:
        Harsh Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
        in ACL 2023
    """
    from flashrag.pipeline import IRCOTPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    print(config["generator_model_path"])
    pipeline = IRCOTPipeline(config, max_iter=5)

    result = pipeline.run(test_data)


def research(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import ReSearchPipeline
    pipeline = ReSearchPipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)


def simplerl(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SimpleRLPipeline
    pipeline = SimpleRLPipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)

def prime(args, config_dict):
    # Eurus-2-7B-PRIME is trained using PRIME (Process Reinforcement through IMplicit rEward) method,
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import PRIMEPipeline
    pipeline = PRIMEPipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)

def ikea(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import IKEAPipeline
    pipeline = IKEAPipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)


def searchr1(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SearchR1Pipeline
    pipeline = SearchR1Pipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)


def torl(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import ToRLPipeline
    pipeline = ToRLPipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)


def autotir(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import AutoTIRPipeline
    pipeline = AutoTIRPipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--config_path", type=str, default="./eval_config.yaml")
    parser.add_argument("--method_name", type=str, default="naive")
    parser.add_argument("--data_dir", type=str, default="/share/project/weiyifan/AutoTIR/data/")
    parser.add_argument("--dataset_name", type=str, default="AIME25")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--save_dir", type=str, default="eval_results/")
    parser.add_argument("--save_note", type=str, default='')
    parser.add_argument("--sgl_remote_url", type=str,
                        default="http://:83") 
    parser.add_argument("--remote_retriever_url", type=str, default="http://:80")
    parser.add_argument("--sandbox_url", type=str, default="http://:81")
    parser.add_argument("--generator_model", type=str,
                        default="/share/project/weiyifan/downloads/trained_ckpts/")
    parser.add_argument("--apply_chat", type=bool, default=True)

    func_dict = {
        "naive": naive,
        "zero-shot": zero_shot,
        "iterretgen": iterretgen,
        "ircot": ircot,
        "research": research,
        "prime": prime,
        "simplerl": simplerl,
        "ikea": ikea,
        "searchr1": searchr1,
        "torl": torl,
        "autotir": autotir,
        "ours": autotir,
    }

    args = parser.parse_args()

    config_dict = {
        "data_dir": args.data_dir,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "save_dir": args.save_dir,
        "save_note": args.save_note if args.save_note else args.method_name,
        "sgl_remote_url": args.sgl_remote_url,
        "remote_retriever_url": args.remote_retriever_url,
        "generator_model": args.generator_model,
        "sandbox_url": args.sandbox_url,
    }

    func = func_dict[args.method_name]
    func(args, config_dict)
