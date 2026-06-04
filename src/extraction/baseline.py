from itertools import combinations

from tqdm import tqdm

import spacy
import networkx as nx


ENTITY_TYPES = {"DATE", "EVENT", "LOC", "MONEY", "ORG", "PERSON", "PRODUCT", "TIME"}

def get_verb(first_entity, second_entity):
    """
    Heuristic: two entities are connected by a verb

    If the token is not a verb the token entities are not connected or are connected in a complex way outside the scope of the heuristic
    """

    # root tokens of both entities
    r1 = first_entity.root
    r2 = second_entity.root
    
    # paths to the top of the sentence dependency tree
    path1 = list(r1.ancestors)
    path2 = list(r2.ancestors)
    
    # 3. Find the lowest common ancestor
    for token in path1:
        if token in path2:
            if token.pos_ == "VERB":
                return token.lemma_ # Base form of the token, with no inflectional suffixes
            return None 
            
    return None


def extract_graph(nlp, emails: list[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for email in tqdm(emails, desc="extracing emails", unit="email"):
        body = email["content"]
        doc = nlp(body)

        for sentence in doc.sents:
            entities = [e for e in sentence.ents if e.label_ in ENTITY_TYPES] #label_ gives the span label and not the hash
            if len(entities) < 2:
                continue # there is no relationship to extract

            for e1, e2 in combinations(entities, 2):
                
                verb = get_verb(e1, e2)
                
                if verb:
                    graph.add_node(e1.text, evidence=e1.text)
                    graph.add_node(e2.text, evidence=e2.text)
                    
                    graph.add_edge(e1.text, e2.text, label=verb)
    
    return graph