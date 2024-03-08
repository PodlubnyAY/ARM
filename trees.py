from functools import reduce
from queue import Queue
import math
import pandas as pd


def calc_support(s: pd.Series, v):
    return s[s == v].size / s.size

node2str = lambda s:[k+":"+str(v) for k, v in sorted(s.value_counts().items())]
    

def calc_entropy(s: pd.Series):
    iterable = list(map(
        lambda x: (x / s.size) * math.log2(x / s.size), 
        s.value_counts(),
    ))
    return - reduce(
        lambda x, y: x + y, 
        iterable
    )


def _get_orginized_attr(df, target) -> str:
    organized_attribute = ""
    for attr in df.columns.delete(df.columns == target):
        entropy, current_entropy = None, 0
        for value in sorted(set(df[attr])):
            if not isinstance(value, (int, float, complex)):
                    value = f"'{value}'"
            name = f"{attr} == {value}"
            df_m = df.query(name)
            current_entropy += calc_entropy(
                df_m[target]
            ) * df_m.shape[0] / df.shape[0]
        
        if entropy is None or entropy > current_entropy:
            entropy = current_entropy
            organized_attribute = attr
    return organized_attribute


class TreeNode:
    def __init__(self, name, query, columns=None, values=None):
        self.name = name
        self.query = query
        self.columns = [] if columns is None else columns
        self.values = values
        self.edges = []
    
    def __str__(self) -> str:
        return f"{self.query} (sup={self.values:.3f})"


class Tree:
    def __init__(
        self, data, target_column, min_support=0.05, min_confidence=0.9
    ) -> None:
        self.root = TreeNode("root", None)
        self.cursor = self.root
        self._data: pd.DataFrame = data
        self.target_column = target_column
        self._support = min_support
        self._confidence = min_confidence
        self.bracnch_candidates = Queue()
        
    @property
    def data(self):
        return self._data
            
    def growth(self):
        self.bracnch_candidates.put(self.root)
        while not self.bracnch_candidates.empty():
            node: TreeNode = self.bracnch_candidates.get()
            if node.name == "root":
                node_data = self._data
            else:
                node_data = self._data.query(node.query)
                node_data = node_data.drop(columns=node.columns)
            if len(node_data.columns) == 1:
                continue
            if calc_entropy(node_data[self.target_column]) == 0:
                continue
            
            organized_attribute = _get_orginized_attr(
                node_data, self.target_column
            )
            if not organized_attribute:
                continue
            name_template = f"{organized_attribute} == {{value}}"
            for value in sorted(set(node_data[organized_attribute])):
                if not isinstance(value, (int, float, complex)):
                    value = f"'{value}'"
                name = name_template.format(value=value)
                # df_m = node_data.query(name)
                support =  0.15 # calc_support(df_m.iloc[:, -1], value)
                if support < self._support:
                    continue
                if node.query:
                    query = " & ".join([node.query, name])
                else:
                    query = name
                
                new_node = TreeNode(
                    name, query, 
                    node.columns + [organized_attribute], 
                    support,
                )
                node.edges.append(new_node)
                self.bracnch_candidates.put(new_node)
    
    def __str__(self) -> str:
        str_tree = f"Tree for {self.target_column}"
        def nested():
            nonlocal str_tree
            cursor: TreeNode = self.cursor
            if cursor.name != "root":
                v = self.data.query(cursor.query)
                v = v.drop(columns=cursor.columns)[self.target_column]
                v = node2str(v)
                str_tree += " ".join(
                    ["\nif " + str(cursor), 
                    "then", 
                    self.target_column, "=", str(v),
                    "confidence = ..."],
                )
            for node in cursor.edges:
                self.cursor = node
                nested()
                
            self.cursor = cursor
        
        nested()
        return str_tree


def __recursion(tree, str_tree="") -> str:
    cursor: TreeNode = tree.cursor
    if cursor.name == "root":
        str_tree = f"Tree for {tree.target_column}"
    else:
        v = tree.data.query(cursor.query)
        v = v.drop(columns=cursor.columns)[tree.target_column]
        v = node2str(v)
        str_tree += " ".join(
            ["\nif"+str(cursor), 
             "then", 
             tree.target_column, "=", v,
             "confidence = ..."],
        )
    for node in cursor.edges:
        tree.cursor = node
        str_tree += __recursion(tree, str_tree)
        tree.cursor = cursor
        
    return str_tree
        
        
if __name__ == "__main__":
    df = pd.read_excel("./validate_input.xlsx")
    i = df.columns[0]
    t = Tree(df, i)
    t.growth()
    print(t)
    # print(f"Tree for {t.target_column}\n" + str(t))