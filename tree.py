import pandas as pd
from queue import Queue

import metrics


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
        self, data, target_column, min_support=0.05, min_threshold=0.9,
        depth=0, width=0, supposed_root_attribute=None, **kwargs,
    ) -> None:
        self.root = TreeNode("root", None)
        self.cursor = self.root
        self._data: pd.DataFrame = data
        self.target_column = target_column
        self._support = min_support
        self.target_support = metrics.calc_multisupport(data[target_column], 0)
        self.width = width
        self.depth = depth
        self._confidence = min_threshold
        self.bracnch_candidates = Queue()
        self.supposed_antecedence = supposed_root_attribute
        self.attrs = tuple(data.columns)
            
    def growth(self):
        self.bracnch_candidates.put((self.root, 0))  # zero is a depth of the tree
        while not self.bracnch_candidates.empty():
            node, current_depth = self.bracnch_candidates.get()
            node : TreeNode = node
            if node.name == "root":
                node_data = self._data
            else:
                node_data = self._data.query(node.query)
                node_data = node_data.drop(columns=node.columns)
            if len(node_data.columns) == 1:
                continue
            if metrics.calc_entropy(node_data[self.target_column]) == 0:
                continue
            if self.depth > 0 and current_depth >= self.depth:
                continue
            
            attrs = node_data.columns.delete(
                node_data.columns == self.target_column
            )
            if node.name != 'root':
                rang = self.attrs.index(node.name.split(" & ")[-1].split(" == ")[-2])
            else:
                rang = -1
            
            for organized_attribute in attrs:
                if rang > self.attrs.index(organized_attribute):
                    # TODO check columns increasing
                    continue
                name_template = f"{organized_attribute} == {{value}}"
                for value in sorted(set(node_data[organized_attribute])):
                    if self.width > 0 and len(node.edges) >= self.width:
                        break
                    if not isinstance(value, (int, float, complex)):
                        value = f"'{value}'"
                    name = name_template.format(value=value)
                    parsel = " & ".join([node.query, name]) if node.query else name
                    support = metrics.calc_support(
                        parsel,
                        self._data,
                    )
                    if support < self._support:
                        continue
                    
                    df_m = node_data.query(name)
                    confidence = metrics.calc_multisupport(
                        df_m[self.target_column],
                        self._confidence,    
                    )
                    new_node = TreeNode(
                        name, 
                        parsel, 
                        node.columns + [organized_attribute], 
                        support, confidence,
                    )
                    node.edges.append(new_node)
                    self.bracnch_candidates.put((new_node, current_depth + 1))
    
    def get_rules(self) -> pd.DataFrame:
        columns=(
            "antecedents", "consequents", 
            "support", "confidence", "lift",
        )
        data = []
        def nested():
            nonlocal data
            cursor: TreeNode = self.cursor
            if cursor.name != "root":
                for v, c in cursor.confidence.items():
                    if c < self._confidence:
                        continue
                    antecedent = cursor.query.replace(' == ', '=')
                    antecedent = antecedent.replace('\'', '')
                    antecedent = frozenset(antecedent.split(' & '))
                    data.append([
                        antecedent,
                        frozenset((f"{self.target_column}={v}",)),
                        cursor.support, c, 
                        c / self.target_support[v], 
                        
                    ])

            for node in cursor.edges:
                self.cursor = node
                nested()
                
            self.cursor = cursor
        
        nested()
        return pd.DataFrame(data, columns=columns)


if __name__ == '__main__':
    t = Tree(
        pd.read_excel('test_input.xlsx'),
        'Feature5', min_threshold=0.01, min_support=0.01
    )
    t.growth()
    print(t.get_rules().sort_values(by=['support']).to_string())