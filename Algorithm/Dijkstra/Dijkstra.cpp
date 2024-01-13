#include <iostream>
#include <vector>
#include <limits>

using namespace std;

#define INF numeric_limits<int>::max()

// 图的邻接矩阵表示
const int MAX_VERTICES = 100;
int graph[MAX_VERTICES][MAX_VERTICES];

// Dijkstra 算法函数
void dijkstra(int start, int vertices) {
    vector<bool> visited(vertices, false);
    vector<int> dist(vertices, INF);

    dist[start] = 0;

    for (int i = 0; i < vertices - 1; ++i) {
        int min_dist = INF, min_index;

        // 找出未访问的节点中距离起始节点最近的节点
        for (int v = 0; v < vertices; ++v) {
            if (!visited[v] && dist[v] < min_dist) {
                min_dist = dist[v];
                min_index = v;
            }
        }

        visited[min_index] = true;

        // 更新最短路径
        for (int v = 0; v < vertices; ++v) {
            if (!visited[v] && graph[min_index][v] && dist[min_index] != INF &&
                dist[min_index] + graph[min_index][v] < dist[v]) {
                dist[v] = dist[min_index] + graph[min_index][v];
            }
        }
    }

    // 输出最短路径
    cout << "节点到其他节点的最短路径：" << endl;
    for (int i = 0; i < vertices; ++i) {
        cout << start << " 到 " << i << " 的最短距离为：" << dist[i] << endl;
    }
}

int main() {
    // 初始化图的邻接矩阵
    int vertices, edges;
    cout << "请输入节点数和边数：" << endl;
    cin >> vertices >> edges;

    cout << "请输入边的信息（起点 终点 距离）：" << endl;
    for (int i = 0; i < edges; ++i) {
        int start, end, weight;
        cin >> start >> end >> weight;
        graph[start][end] = weight;
        graph[end][start] = weight; // 若为有向图则去掉此行
    }

    int start_node;
    cout << "请输入起始节点：" << endl;
    cin >> start_node;

    dijkstra(start_node, vertices);

    return 0;
}
