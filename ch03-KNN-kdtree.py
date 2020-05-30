import numpy as np



# 参考链接：https://github.com/tsoding/kdtree-in-python

def distance(point1, point2):
    # point为ndarray
    return np.sqrt(np.sum((point1 - point2) ** 2))


class Kdnode:
    def __init__(self, point, left=None, right=None):
        # point为kd树的节点对应的数据点
        self.point = point
        # 左右子树
        self.left = left
        self.right = right


class Kdtree:
    def __init__(self):
        # 初始化的根节点
        self.root = None

    def build_tree(self, X):
        # 根据数据集X建立kd树，X为ndarray，每个元素表示一个数据点
        # 构建过程就是每次选择一个维度，计算所有数据点对应的该维度的值的中位数做分割点
        # 使用所有改维度小于该分割点的数据点构建左子树，其余的数据点构建右子树
        # 下面使build_helper递归构建kd树
        def build_helper(X, cutting_dim=0):
            # 建树辅助函数，用于递归构建kd树，cutting_dim表示当前用于分割的维度的索引
            n = len(X)
            if n == 0:  # 没有数据就返回空节点
                return None
            k = len(X[0])
            # 将数据根据cutting_dim排序
            sorted_points = sorted(X, key=lambda point: point[cutting_dim])
            # 返回新建的节点
            return Kdnode(
                sorted_points[n // 2],
                build_helper(sorted_points[:n // 2], (cutting_dim + 1) % k),
                build_helper(sorted_points[n // 2 + 1:], (cutting_dim + 1) % k)
            )
        self.root = build_helper(X)

    def preorder(self):
        # 先序遍历
        def preorder_helper(node):
            if node is None:
                return
            print(node.point)
            preorder_helper(node.left)
            preorder_helper(node.right)

        preorder_helper(self.root)

    def nearest_point(self, target):
        # 找出离target最近的点
        def closer_distance(pivot, p1, p2):
            # 返回p1，p2中离pivot最近的点
            if p1 is None:
                return p2
            if p2 is None:
                return p1
            d1 = distance(pivot, p1)
            d2 = distance(pivot, p2)
            if d1 < d2:
                return p1
            else:
                return p2

        def nearest_point_helper(node, target, cutting_dim=0):
            # 寻找以node节点为根节点的子树中离target最近的点
            if node is None:
                return None

            # k表示数据维度
            k = len(target)

            if target[cutting_dim] < node.point[cutting_dim]:
                # 目标数据点对应的维度值小于当前节点对应的维度的值，所以转到左子节点
                next_branch = node.left     # 接下来继续搜索的分支节点
                opposite_branch = node.right    # 继续搜索的分支节点的兄弟分支
            else:
                next_branch = node.right
                opposite_branch = node.left

            # 在以next_branch为根节点的子树中寻找离target最近的节点，将得到的最近的节点和当前节点相比，将较近的点作为当前最近的节点
            best = closer_distance(target,
                                   nearest_point_helper(next_branch, target, (cutting_dim + 1) % k),
                                   node.point)
            # 如果目标节点到当前得到的最近节点的距离大于目标节点与当前正在遍历的树节点在切分维度上的距离，则需要搜索当前节点的父节点的另一个子节点
            # 这一步的理解：当前节点有一个切分维度，根据此切分维度的取值将数据点切分
            # 当目标节点到该切分超平面的距离大于目标节点到当前最近的点的距离时，切分超平面的另一面不可能存在更近的点，否则，可能存在更近的点
            if distance(target, best) > abs(target[cutting_dim] - node.point[cutting_dim]):
                best = closer_distance(target,
                                       nearest_point_helper(opposite_branch, target, (cutting_dim + 1) % k),
                                       best)

            return best

        return nearest_point_helper(self.root, target, 0)


if __name__ == "__main__":
    X = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    tree = Kdtree()
    tree.build_tree(X)
    print("kd树先序遍历结果")
    tree.preorder()

    target = np.array([3, 4.5])
    print("离(3, 4.5)最近的点为" + str(tree.nearest_point(target)))
    for i in range(len(X)):
        print(X[i], distance(target, X[i]))