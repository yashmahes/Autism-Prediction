class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BST(object):
    def __init__(self):
        self.root = None

    def insert(self, value):
        node = Node(value)

        if self.root == None:
            self.root = node 

        else:
            current = self.root 

            while current != None:

                if value < current.value:
                    if current.left == None:
                        current.left = node
                        break
                    current = current.left


                elif value > current.value:
                    if current.right == None:
                        current.right = node 
                        break

                    current = current.right


    def find(self, value):

        current = self.root 

        while current != None:

            if value == current.value:
                return True

            elif value < current.value:
                current = current.left 

            else:
                current = current.right 

        return False


    def traverse_pre_order(self, node):

        if node == None:
            return

        print(node.value)
        self.traverse_pre_order(node.left)
        self.traverse_pre_order(node.right)

    def traverse_in_order(self, node):

        if node == None:
            return

        self.traverse_in_order(node.left)
        print(node.value)

        self.traverse_in_order(node.right)


    def traverse_post_order(self, node):

        if node == None:
            return

        self.traverse_post_order(node.left)

        self.traverse_post_order(node.right)

        print(node.value)


    def height(self, node):
        if node == None:
            return -1
        if node.left == None and node.right == None:
            return 0

        return 1 + max(self.height(node.left), self.height(node.right))


    def minimum_value(self, node):
        if node == None:
            return -1

        if node.left == None and node.right == None:
            return node.value

        return min(node.value, min(self.minimum_value(node.left), self.minimum_value(node.right)))


    
    def equals_to(self, other):

        return self.equals(self.root, other.root)

    def equals(self, root1, root2):

        if root1 == None and root2 == None:
            return True 
        elif root1 == None and root2 != None:
            return False 

        elif root1 != None and root2 == None:
            return False

        else:
            return root1.value == root2.value and self.equals(root1.left, root2.left) and self.equals(root1.right, root2.right)

    


    def is_bst(self):

        return self.validate(self.root, -10000000, 10000000)

    def validate(self, node, min_, max_):
        if node == None:
            return True

        return max_ < node.value > min_ and self.validate(node.left, min_, node.value) and self.validate(node, node.value, max_)


    def get_nodes_at_distance(self, k):

        if self.root == None:
            return 

        arr = []
        self.distance(self.root, k, arr)
        return arr

    def distance(self, node, k, arr):

        if node == None:
            return arr

        if k == 0:
            arr.append(node.value)

        else:
            self.distance(node.left, k-1, arr)
            self.distance(node.right, k-1, arr)

        return arr

    def level_order_traversal(self):
        for i in range(3):
            arr = self.get_nodes_at_distance(i)
            print(arr)

tree = BST()
tree.insert(7)
tree.insert(4)
tree.insert(9)
tree.insert(1)
tree.insert(6)
tree.insert(8)
tree.insert(10)

# tree.traverse_in_order(tree.root)

# print(tree.height(tree.root))

arr = tree.get_nodes_at_distance(2)
print(arr)

tree.level_order_traversal()