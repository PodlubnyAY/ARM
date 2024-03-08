import math
import pandas as pd
from functools import reduce
from multiprocessing import Pool


df0 = pd.read_excel("./validate_input.xlsx")
cstr = lambda s:[k+":"+str(v) for k, v in sorted(s.value_counts().items())]
entropy = lambda s:-reduce(lambda x, y:x+y, map(lambda x:(x/len(s))*math.log2(x/len(s)), s.value_counts()))
support = lambda s, v: s[s==v].size / s.size
 

def exceptioner(f ):
    def wrapped(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            print(f"ERROR {f.__name__}{args} {e}")
    return wrapped


def growth_tree(target_index):
    # Структура данных Decision Tree
    df = df0.astype(str)
    target_column = df0.columns[target_index]
    # df = df0.drop(columns=target_column).astype(str)
    # df = df.join(df0[target_column].astype(str))
    tree = {
        # name: Название этого нода (узла)
        "name":"root",  # "decision tree "+df0.columns[-1]+" "+str(cstr(df0.iloc[:, -1])), 
        # df: Данные, связанные с этим нодом (узлом)
        "df":df, 
        # edges: Список ребер (ветвей), выходящих из этого узла, 
        # или пустой массив, если ниже нет листового узла.
        "edges":[], 
    }

    # Генерацию дерева, у узлов которого могут быть ветви, сохраняем в open
    open = [tree]
    # Зацикливаем, пока open не станет пустым
    while(len(open)!=0):
        # Вытаскиваем из массива open первый элемент, 
        # и вытаскиваем данные, хранящиеся в этом узле
        node = open.pop(0)
        df_node: pd.DataFrame = node["df"]
        # В случае, если энтропия этого узла равна 0, мы больше не можем вырастить из него новые ветви
        # поэтому прекращаем ветвление от этого узла
        if entropy(df_node[target_column]) == 0:
            continue
        # Создаем переменную, в которую будем сохранять список значений атрибута с возможностью разветвления
        branching_attrs = {}
        # Исследуем все атрибуты, кроме последнего столбца класса атрибутов
        for attr in df_node.columns.delete(df_node.columns == target_column):
            # Создаем переменную, которая хранит значение энтропии при ветвлении с этим атрибутом, 
            # данные после разветвления и значение атрибута, который разветвляется.
            branching_attrs[attr] = {
                "entropy": 0, 
                "dfs": [], 
                "values": [],
            }
            # Исследуем все возможные значения этого атрибута. 
            # Кроме того, sorted предназначен для предотвращения изменения порядка массива, 
            # из которого были удалены повторяющиеся значения атрибутов, при каждом его выполнении.
            for value in sorted(set(df_node[attr])):
                # Фильтруем данные по значению атрибута
                df_m = df_node.query(attr+"=='"+value+"'")
                # Высчитываем энтропию, данные и значения сохрнаяем
                branching_attrs[attr]["entropy"] += entropy(df_m[target_column]) * df_m.shape[0] / df_node.shape[0]
                branching_attrs[attr]["dfs"] += [df_m]  # Bad for memory. Better to store indexes for this
                branching_attrs[attr]["values"] += [value]

        # Если не осталось ни одного атрибута, значение которого можно разделить, 
        # прерываем исследование этого узла.
        if len(branching_attrs) == 0:
            continue
        # Получаем атрибут с наименьшим значением энтропии
        attr = min(branching_attrs, key=lambda x:branching_attrs[x]["entropy"])
        # Добавляем каждое значение разветвленного атрибута
        # и данные, полученные после разветвления, в наше дерево и в open.
        for d, v in zip(branching_attrs[attr]["dfs"], branching_attrs[attr]["values"]):
            m = {
                "name": attr + "=" + v, 
                "edges":[], 
                "df": d.drop(columns=attr),
            }
            node["edges"].append(m)
            open.append(m)
    return tree
        

def print_tree(tree, target, tree_string=""):
    tree_string += f"'{tree['name']}'"
    if tree['name'] == 'root':
        print(tree["df"].iloc[:, target].name)
        tree_string = "if "
    else:
        tree_string += " and "
    lines = []
    for e in tree["edges"]:
        lines.append(print_tree(e, target, tree_string))
    if not lines:
        print(tree_string[:-5] + " then " + str(cstr(tree["df"].iloc[:, -1])))

if __name__ == "__main__":
    
    # with Pool(5) as p:
    #     forest = p.map(growth_tree, range(len(df0.columns)))
        
    # for tree in forest:
    #     print_tree(tree)
    i=0
    # for i in range(len(df0.columns)):
    tree = growth_tree(i)
    print_tree(tree, i)
    
