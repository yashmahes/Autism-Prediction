class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BST:
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

                else:
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

            elif value > current.value:
                current = current.right 

        return False 


        

                






