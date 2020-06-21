#Author Giovanni Fabbretti

import os
import pandas as pd
import networkx as nx

def build_empty_table(path, output):
    assert os.path.exists(path)

    if(os.path.exists(output)):
        answer = input("Sembra che una tabella esista gia', vuoi sovrascriverla?\n")
        if(answer == "N"):
            return

    graph_info = pd.read_csv(path)

    landmark_list = [row["POSITION"] for _,row in graph_info.loc[graph_info["Name"] == "object landmark"].iterrows()]

    print(landmark_list)

    df = pd.DataFrame({}, columns =['object'] + landmark_list)

    df.to_csv(output, encoding='utf-8')

def add_row_to_dataframe(path):
    assert os.path.exists(path)

    df = pd.read_csv(path, index_col=0)

    keys = list(df.columns)[1:]

    values = []

    new_object = input("Qual e' il nome dell'oggetto da inserire?\n")

    for key in keys:
        while True:
            value = input("Quante volte l'oggetto si e' trovato in questa posizione?\n" + str(key) +": ")
            if(value.isnumeric() and int(value) >= 0):
               break
        values += [value]

    df.loc[len(df)] = [new_object] + values

    df.to_csv(path)

#build_empty_table("graph.csv", "table.csv")
#add_row_to_dataframe('./table.csv')

def build_graph(path):
    assert os.path.exists(path)

    graph_info = pd.read_csv(path)

    g = nx.Graph()

    #adding pose nodes to the graph
    for _, row in graph_info.loc[graph_info["Name"] == "pose landmark"].iterrows():
        x = row["Position X"]
        y = row["Position Y"]
        position = row["POSITION"]

        g.add_node(("pose",x,y,position))

        #adding object landmark to the graph
    for _, row in graph_info.loc[graph_info["Name"] == "object landmark"].iterrows():
        x = row["Position X"]
        y = row["Position Y"]
        position = row["POSITION"]

        g.add_node(("object",x,y,position))

    #adding edges to the graph
    for _, row in graph_info.loc[graph_info["Name"] == "Line"].iterrows():
        start_x = row["Start X"]
        start_y = row["Start Y"]

        end_x = row["End X"]
        end_y = row["End Y"]

        distance = row["Length"]

        node1 = ""
        node2 = ""

        for label,x,y,position in g.nodes():
            if x == start_x and y == start_y:
                node1 = (label, x, y, position)

            if x == end_x and y == end_y:
                node2 = (label, x, y, position)


        assert node1 != ""
        assert node2 != ""

        g.add_edge(node1, node2, weight=distance)

    return g


def display_probs(d):
    for key, value in d.items():
        print(f'Place: {key:8} Prevalence: {100*value:.2f}%.')

