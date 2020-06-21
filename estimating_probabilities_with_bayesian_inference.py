import pandas as pd
import numpy as np
import os
import pymc3 as pm
import networkx as nx
import json
import speech_recognition as sr


# Helper functions
from utils import display_probs, build_graph, add_row_to_dataframe

with open("./credentials_speech_to_text_google.json") as f:
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = json.dumps(json.load(f))


def get_distances(path, source):
    assert os.path.exists(path)

    g = build_graph(path)
    distances = nx.shortest_path_length(g, source=source, weight="weight" )

    distances_filtered = dict()

    for key in distances.keys():
        _,_,_,position = key
        distances_filtered[position] = distances[key]

    return distances_filtered

def check_similar_objects(df, object_to_search):

    max = 0
    object_most_similar = None

    try:
        similars = gmodel.most_similar(positive=[object_to_search], topn=10)
    except KeyError: #object not present in the table
        return None

    similars = [x for x,_ in similars]

    for similar in similars:
        print(similar)
        if(len(df.loc[df["object"] == similar]) > max):
            object_most_similar = similar
            max = sum(df.loc[df["object"] == similar].drop("object",1).values[0])

    return object_most_similar


def get_object_places(table_path, graph_path):
    assert os.path.exists(table_path)
    assert os.path.exists(graph_path)

    df = pd.read_csv(table_path, index_col = 0)

    while True:
        #speech to text part
        # obtain audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:

            print("Quale oggetto stai cercando?\n")
            #audio = r.listen(source)

            # recognize speech using Google Cloud Speech
            try:
                #object_to_search = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS).strip()
                object_to_search = "milk"
                similar = check_similar_objects(df, object_to_search)
                print(similar)
                if(len(df.loc[df["object"] == object_to_search]) > 0 ):
                    break
                elif(similar != None):
                    answer = input(object_to_search + " non trovato, trovata compatibilita' con " + similar +", utilizzare la sua distribuzione?[Y/n]\n")
                    if(answer == "Y"):
                        object_to_search = similar
                        break

            except sr.UnknownValueError:
                print("Google Cloud Speech could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Cloud Speech service; {0}".format(e))

    row = df.loc[df["object"] == object_to_search].drop('object', 1)

    for x in list(zip(row.keys() ,row.values[0])):
        print(str(x[0]) + " " + str(x[1]))

    places = row.keys()
    knowledge = row.values[0]
    number_of_places = len(knowledge)

    distances_dict = get_distances(graph_path, ("pose",7.3533,2.2700,"corridor-2"))
    distances = [ distances_dict[key] for key in places]
    max_distance = max(distances)

    print("Sommatoria = " + str(sum(distances)) + " numero osservazioni = " + str(sum(knowledge)) + "Rapporto S/o =" + str(sum(distances)/sum(knowledge)))

    inverted_distances = list(map(lambda x: abs(x-max_distance+1)/3, distances))

    prior_knowledge = np.array(inverted_distances)

    with pm.Model() as model:
        # Parameters of the Multinomial are from a Dirichlet
        parameters = pm.Dirichlet('parameters', a=prior_knowledge, shape=number_of_places)
        # Observed data is from a Multinomial distribution
        observed_data = pm.Multinomial(
            'observed_data', n=sum(knowledge), p=parameters, shape=number_of_places, observed=knowledge)

    with model:
        # Sample from the posterior
        trace = pm.sample(draws=1000, chains=2, tune=500,
                          discard_tuned_samples=True)

        trace_df = pd.DataFrame(trace['parameters'], columns = places)


    # For probabilities use samples after burn in
    pvals = trace_df.iloc[:, :number_of_places].mean(axis = 0)
    display_probs(dict(zip(places, pvals)))

get_object_places("table.csv", "graph.csv")
#add_row_to_dataframe("table.csv")




