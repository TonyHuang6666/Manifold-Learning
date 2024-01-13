import heapq  # 导入 heapq 模块，用于堆操作

class Graph:  # 定义 Graph 类
    def __init__(self):  # 初始化方法
        self.nodes = set()  # 初始化节点集合
        self.edges = {}  # 初始化边字典

    def add_node(self, value):  # 添加节点方法
        self.nodes.add(value)  # 将节点添加到节点集合中
        self.edges[value] = []  # 初始化该节点的边为空列表

    def add_edge(self, from_node, to_node, weight):  # 添加边的方法
        self.edges[from_node].append((to_node, weight))  # 添加从 from_node 到 to_node 的边
        self.edges[to_node].append((from_node, weight))  # 添加从 to_node 到 from_node 的边

    def dijkstra(self, start):  # Dijkstra 算法实现
        distances = {node: float('infinity') for node in self.nodes}  # 初始化节点到起点的距离为无穷大
        distances[start] = 0  # 起点到自身的距离为 0
        visited = set()  # 初始化已访问节点的集合为空集合
        priority_queue = [(0, start)]  # 优先队列初始化为包含起点的元组列表

        while priority_queue:  # 循环直到优先队列为空
            current_distance, current_node = heapq.heappop(priority_queue)  # 从优先队列中弹出距离最小的节点
            visited.add(current_node)  # 将该节点标记为已访问

            for neighbor, weight in self.edges[current_node]:  # 遍历当前节点的所有邻居节点
                if neighbor not in visited:  # 如果邻居节点未被访问过
                    new_distance = current_distance + weight  # 计算新的距离
                    if new_distance < distances[neighbor]:  # 如果新的距离小于目前记录的距离
                        distances[neighbor] = new_distance  # 更新距离字典
                        heapq.heappush(priority_queue, (new_distance, neighbor))  # 将新距离和邻居节点加入优先队列
        return distances  # 返回节点到起点的最短距离

# 创建图实例
graph = Graph()  # 实例化 Graph 类
graph.add_node("A")  # 添加节点 "A"
graph.add_node("B")  # 添加节点 "B"
graph.add_node("C")  # 添加节点 "C"
graph.add_node("D")  # 添加节点 "D"
graph.add_node("E")  # 添加节点 "E"

# 添加边以及权重
graph.add_edge("A", "B", 6)  # 添加从节点 "A" 到节点 "B" 的边，权重为 6
graph.add_edge("A", "D", 1)  # 添加从节点 "A" 到节点 "D" 的边，权重为 1
graph.add_edge("B", "D", 2)  # 添加从节点 "B" 到节点 "D" 的边，权重为 2
graph.add_edge("B", "E", 2)  # 添加从节点 "B" 到节点 "E" 的边，权重为 2
graph.add_edge("B", "C", 5)  # 添加从节点 "B" 到节点 "C" 的边，权重为 5
graph.add_edge("C", "E", 5)  # 添加从节点 "C" 到节点 "E" 的边，权重为 5
graph.add_edge("D", "E", 1)  # 添加从节点 "D" 到节点 "E" 的边，权重为 1

# 计算从节点 "A" 到其他节点的最短路径
shortest_paths = graph.dijkstra("B")  # 使用 Dijkstra 算法计算最短路径
print(shortest_paths)  # 打印从节点 "A" 到其他节点的最短路径长度
