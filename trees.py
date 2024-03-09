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
    def __init__(self, name, query, columns=None, support=None, confidence=None):
        self.name = name
        self.query = query
        self.columns = [] if columns is None else columns
        self.support = support
        self.confidence = confidence
        self.edges = []
    
    def get_rules(self, target_name) -> str:
        rules = ""
        for v, c in self.confidence.items():
            rules += " ".join([
                "\nif", self.query,
                f"(sup={self.support:.3f})",
                "then", target_name, "=", str(v),
                f"(confidence={c:.3f})",
            ])
            
        return rules


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
    
    def calc_confidence(self, s: pd.Series):
        confidences = {}
        for value, count in s.value_counts().items():
            confidence = count / s.size
            if confidence > self._confidence:
                confidences[value] = confidence
        
        return confidences
            
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
                support = calc_support(node_data[organized_attribute], value)
                if support < self._support:
                    continue
                
                if not isinstance(value, (int, float, complex)):
                    value = f"'{value}'"
                
                name = name_template.format(value=value)
                df_m = node_data.query(name)
                confidence = self.calc_confidence(df_m[self.target_column])
                new_node = TreeNode(
                    name, 
                    " & ".join([node.query, name]) if node.query else name, 
                    node.columns + [organized_attribute], 
                    support, confidence,
                )
                node.edges.append(new_node)
                self.bracnch_candidates.put(new_node)
    
    def get_rules(self) -> pd.DataFrame:
        columns=(
            "consequents", "antecedents", 
            "support", "confidence",
        )
        data = []
        def nested():
            nonlocal data
            cursor: TreeNode = self.cursor
            if cursor.name != "root":
                for v, c in cursor.confidence.items():
                    if c < self._confidence:
                        continue
                    data.append([
                        cursor.query,
                        f"{self.target_column}=={v}",
                        cursor.support, c,
                    ])

            for node in cursor.edges:
                self.cursor = node
                nested()
                
            self.cursor = cursor
        
        nested()
        return pd.DataFrame(data, columns=columns)
        
        
if __name__ == "__main__":
    df = pd.read_excel("./validate_input.xlsx")
    i = df.columns[0]
    t = Tree(df, i, min_confidence=0.55)
    t.growth()
    rules = t.get_rules().round(3)
    print(rules.to_string(index=False))
    