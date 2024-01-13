#include <iostream>
#include <vector>
#include <limits>

using namespace std;

#define INF numeric_limits<int>::max()

// ͼ���ڽӾ����ʾ
const int MAX_VERTICES = 100;
int graph[MAX_VERTICES][MAX_VERTICES];

// Dijkstra �㷨����
void dijkstra(int start, int vertices) {
    vector<bool> visited(vertices, false);
    vector<int> dist(vertices, INF);

    dist[start] = 0;

    for (int i = 0; i < vertices - 1; ++i) {
        int min_dist = INF, min_index;

        // �ҳ�δ���ʵĽڵ��о�����ʼ�ڵ�����Ľڵ�
        for (int v = 0; v < vertices; ++v) {
            if (!visited[v] && dist[v] < min_dist) {
                min_dist = dist[v];
                min_index = v;
            }
        }

        visited[min_index] = true;

        // �������·��
        for (int v = 0; v < vertices; ++v) {
            if (!visited[v] && graph[min_index][v] && dist[min_index] != INF &&
                dist[min_index] + graph[min_index][v] < dist[v]) {
                dist[v] = dist[min_index] + graph[min_index][v];
            }
        }
    }

    // ������·��
    cout << "�ڵ㵽�����ڵ�����·����" << endl;
    for (int i = 0; i < vertices; ++i) {
        cout << start << " �� " << i << " ����̾���Ϊ��" << dist[i] << endl;
    }
}

int main() {
    // ��ʼ��ͼ���ڽӾ���
    int vertices, edges;
    cout << "������ڵ����ͱ�����" << endl;
    cin >> vertices >> edges;

    cout << "������ߵ���Ϣ����� �յ� ���룩��" << endl;
    for (int i = 0; i < edges; ++i) {
        int start, end, weight;
        cin >> start >> end >> weight;
        graph[start][end] = weight;
        graph[end][start] = weight; // ��Ϊ����ͼ��ȥ������
    }

    int start_node;
    cout << "��������ʼ�ڵ㣺" << endl;
    cin >> start_node;

    dijkstra(start_node, vertices);

    return 0;
}
