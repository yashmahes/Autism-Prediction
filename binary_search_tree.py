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
            return 

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
            if current.value == value:
                return True 

            elif value < current.value:
                current = current.left 

            elif value > current.value:
                current = current.right 


        return False


    def taverse_in_order(self, node):

        if node != None:

            self.taverse_in_order(node.left)
            print(node.value)
            self.taverse_in_order(node.right)

    def taverse_pre_order(self, node):

        if node != None:
            print(node.value)

            self.taverse_pre_order(node.left)
            self.taverse_pre_order(node.right)

    def taverse_post_order(self, node):

        if node != None:

            self.taverse_post_order(node.left)
            self.taverse_post_order(node.right)
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



    def is_bst(self):

        return self.validate(self.root, -1000000, 1000000)


    def validate(self, node, min_value, max_value):

        if node == None:
            return True


        elif node.left != None and node.right != None:
            return min_value < node.value < max_value and self.validate(node.left, min_value, node.value) and self.validate(node.right, node.value, max_value)



