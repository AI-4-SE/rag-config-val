from cfgnet.network.network import Network, NetworkConfiguration
from sklearn.model_selection import train_test_split
import pandas as pd
import random

EVAL_REPOS = [
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/piggymetrics",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/pig",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/litemall",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/mall",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/mall-swarm",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/Spring-Cloud-Platform",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/jetlinks-community",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/music-website",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/spring-boot-admin",
    "/home/simisimon/GitHub/cval_evaluation/spring-projects/apollo",
]


def get_dependencies(repo_dir: str, num_dependencies: int = 50) -> None:

    repo_name = repo_dir.split("/")[-1]

    cfg = NetworkConfiguration(
        project_root_abs=repo_dir,
        enable_internal_links=True,
        enable_all_conflicts=True,
        enable_static_blacklist=False
    )

    network = Network.init_network(cfg=cfg)

    links = network.links

    sampled_links = random.choices(list(links), k=num_dependencies)

    link_list = []

    for link in sampled_links:
        link_list.append({
            "dependency_category": "value-equality",
            "link_str": str(link),
            "project": repo_name,
            "option_name": link.node_a.get_options(),
            "option_value": link.node_a.name,
            "option_type": link.node_a.config_type,
            "option_file": link.artifact_a.name,
            "option_technology": link.artifact_a.concept_name,
            "dependent_option_name": link.node_b.get_options(),
            "dependent_option_value": link.node_b.name,
            "dependent_option_type": link.node_b.config_type,
            "dependent_option_file": link.artifact_b.name,
            "dependent_option_technology": link.artifact_b.concept_name 
        })


    df = pd.DataFrame(data=link_list)
    df.to_csv(f"../data/projects/{repo_name}_dependencies.csv", index=False)


def stratified_sampling():
    data_file = "../data/evaluation/all_dependencies.csv"

    df = pd.read_csv(data_file)

    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["project"], random_state=42)

    train_df.to_csv("../data/evaluation/train_dependencies.csv", index=False)
    test_df.to_csv("../data/evaluation/test_dependencies.csv", index=False)


if __name__ == "__main__":
    #print("Get Repositories")
    #for repo_dir in EVAL_REPOS:
    #    get_dependencies(repo_dir=repo_dir, num_dependencies=10)

    stratified_sampling()