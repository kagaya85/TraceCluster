from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Iterable, Iterator, Optional, TypeVar

from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets

from matplotlib.colors import hsv_to_rgb

# data type
T = TypeVar("T")


def distance(x: T, y: T) -> float:
    return np.linalg.norm(x - y)


def update_center(centre: T, count: int, data_sample: T) -> T:
    return ((count - 1) * centre + data_sample) / count


@dataclass
class MicroCluster(Generic[T]):
    centre: T
    energy: float = 1
    count: int = 1
    edges: set[MicroCluster[T]] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.energy)

    def __eq__(self, o: object) -> bool:
        xd = super().__eq__(o)
        return xd


class CEDAS(Generic[T]):
    def __init__(
        self,
        stream: Iterator[T],    # data stream
        # 0. Parameter Selection
        r0: float,    # radius
        decay: float,    # a time based decay value
        threshold: int,    # expert knowledge
    ) -> None:
        self.stream = stream
        self.r0 = r0
        self.decay = decay
        self.micro_clusters: MicroCluster = []
        self.changed_cluster: Optional[MicroCluster] = None
        self.threshold = threshold

    # 1. Initialization
    def initialization(self):
        first_sample = self.stream[0]
        self.stream = self.stream[1:]
        first_cluster = MicroCluster(first_sample)
        self.micro_clusters: list[MicroCluster] = [first_cluster]

    # 2. Update Micro-Clusters
    def update(self, data_sample: T) -> None:
        # find nearest micro cluster
        nearest_cluster = min(
            self.micro_clusters,
            key=lambda cluster: distance(data_sample, cluster.centre),
        )
        min_dist = distance(data_sample, nearest_cluster.centre)

        if min_dist < self.r0:
            nearest_cluster.energy = 1
            nearest_cluster.count += 1

            # if data is within the kernel?
            if min_dist < self.r0 / 2:
                # todo: xd
                nearest_cluster.centre = update_center(
                    nearest_cluster.centre, nearest_cluster.count, data_sample
                )

            self.changed_cluster = nearest_cluster
        else:
            new_micro_cluster = MicroCluster(data_sample)
            self.micro_clusters.append(new_micro_cluster)

    # 3. Kill Clusters
    def kill(self) -> None:
        for i, cluster in enumerate(self.micro_clusters):
            cluster.energy -= self.decay    # 这里的衰减方式是不是得改一下，用衰减窗口

            if cluster.energy < 0:
                # Remove all edges containing the micro-cluster
                for c in self.micro_clusters:
                    c.edges.discard(cluster)

                self.changed_cluster = cluster
                self.update_graph()

                # remove cluster
                del self.micro_clusters[i]

    # 4. Update Cluster Graph
    def update_graph(self) -> None:
        if self.changed_cluster and self.changed_cluster.count > self.threshold:
            # find neighbors
            neighbors = {
                cluster
                for cluster in self.micro_clusters
                if distance(cluster.centre, self.changed_cluster.centre)
                <= 1.8 * self.r0
                and cluster.count > self.threshold
            }
            self.changed_cluster.edges = neighbors

            for cluster in neighbors:
                cluster == self.changed_cluster
                cluster.edges.add(self.changed_cluster)

            for cluster in self.micro_clusters:
                if self.changed_cluster in cluster.edges and cluster not in neighbors:
                    cluster.edges.remove(self.changed_cluster)




    def run(self) -> None:
        self.initialization()

        for data_sample in self.stream:
            self.changed_cluster = None
            self.update(data_sample)
            self.kill()

            if self.changed_cluster and self.changed_cluster.count > self.threshold:
                self.update_graph()





    def get_macro_cluster(self) -> list[set[MicroCluster]]:
        seen: set[MicroCluster] = set()

        def dfs(cluster) -> set[MicroCluster]:
            seen.add(cluster)
            return {cluster}.union(
                *map(dfs, [edge for edge in cluster.edges if edge not in seen])
            )

        result = []

        for cluster in self.micro_clusters:
            if cluster.count > self.threshold:
                if cluster not in seen:
                    result.append(dfs(cluster))

        return result


if __name__ == "__main__":
    # data = np.genfromtxt("data.csv", delimiter=",")

    datasets = [
        {
            "data": datasets.make_circles(n_samples=1500, factor=0.5, noise=0.05),
            "xlim": [-1.5, 1.5],
            "ylim": [-1.5, 1.5],
            "r": 0.19,
        },
        {
            "data": datasets.make_moons(n_samples=1500, noise=0.05),
            "xlim": [-1.5, 2.5],
            "ylim": [-1.0, 1.5],
            "r": 0.18,
        },
        {
            "data": datasets.make_blobs(n_samples=1500, random_state=8),
            "xlim": [-10.0, 12.0],
            "ylim": [-15.0, 15.0],
            "r": 0.5,
        },
    ]

    for dataset in datasets:
        data = dataset["data"][0]

        cedas = CEDAS(
            data,
            r0=dataset["r"],
            decay=0.001,
            threshold=5,
        )
        cedas.run()

        for i, macro in enumerate(cedas.get_macro_cluster()):
            color = hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1])
            for cluster in macro:
                cluster.color = color

        fig = plt.figure()

        plt.scatter(data.T[0], data.T[1], marker=".", color="black")

        for cluster in cedas.micro_clusters:
            if cluster.count > cedas.threshold:
                plt.gca().add_patch(
                    plt.Circle(
                        (cluster.centre[0], cluster.centre[1]),
                        cedas.r0,
                        alpha=0.4,
                        color=cluster.color,
                    )
                )

        plt.xlim(dataset["xlim"])
        plt.ylim(dataset["ylim"])
        plt.show()
