from __future__ import unicode_literals
import spacy
from spacy import displacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from spacy.tokens import Doc
from collections import Counter
import numpy
import en_core_web_sm
import collections

nlp = en_core_web_sm.load()
'''
#Model
en_core_web_sm

#Steps
#step 1: PRONOUN cleaning
#	  1.1 replace named person with "person"
#         example: Mary Jane with daughters --> a person with daughters
#	  1.2 remove DATE, TIME, ORG, GPE, LOC, PRODUCT, EVENT, WORK ART, LAW, LANGUAGE, FAC, NORP, PERCENT, MONEY, QUANTITY, ORIGINAL  	
#		and get their parent node, if there's no parent node, then abort this sentence)
#         example: A women group is photographed at the Ashley Power Plant up Taylor Mountain road --> A women group is photographed
#step 2: Parent nodes cleaning 
#	  2.1 if the parent node is 'DET', 'NOUN', 'NUM', 'PUNCT', 'CCONJ', 'VERB', remove the subtrees nodes of the lefts if it's not in the list 
#	  2.2 if the parent node is PREP or other types, remove the nodes and their children nodes
#         example: A woman tends to a fire at a Ute Indian camp near Brush Creek. --> A woman tends to a fire at a camp .

#step 3: NOUN entities cleaning
#	  3.1 replace the chunk of non-named enties chunk with simple entity along with CC, CD, DT and IN
#	  3.2 replace CD type number with string value, e.g. 3, three
#         example: Ute couple with child stand in front of old cars --> couple with child stand in front of cars

#step 4: Reform
#	  4.1 put person entity and chunk replacement in position
#	  4.2 replace person entities with understandable words, e.g. ,if multiple people then use num people instead, e.g. "a person" "two people", else say "a group of people"
#	  4.3 replace nums of person entities (>=3) replace with "a group of " + noun
#	  4.4 remove the tokens with index in indexes list of the sentence and convert the array to sentence 

'''


# function to delete specified indexed token from the str
def remove_span(doc, indexes):
    np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
    np_array = numpy.delete(np_array, indexes, axis=0)
    doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
    doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
    return doc2


# function to add index in all subtree matching the criteria
def loop_tree(Token, indexes):
    #   for noun related type token then keep the node along with its children node
    if Token.pos_ in ('DET', 'NOUN', 'NUM', 'PUNCT', 'CCONJ'):
        #   remove lefts and subtrees of children
        for child in Token.children:
            for c in child.subtree:
                if c.pos_ not in ('DET', 'NOUN', 'NUM', 'PUNCT', 'CCONJ', 'VERB'):
                    if c.i not in indexes:
                        indexes.append(c.i)
    else:
        for c in Token.subtree:
            if c.i not in indexes:
                indexes.append(c.i)
    return


#   function to replace person names in the document with "person"
def replace_person_name(people_list, update_list, indexes):
    for person in people_list:
        update_list[person.start] = "person"
        for i in range(person.start + 1, person.end):
            if i not in indexes:
                indexes.append(i)
    return


#   function to replace chunk with shortened term
def replace_chunk_name(chunk_list, indexes):
    for chunk in chunk_list:
        for child in chunk.root.lefts:
            if child.pos_ not in ('DET', 'NOUN', 'NUM') and child.i not in indexes:
                indexes.append(child.i)
    return


#   function to integrate sentence into understandable phrase
def reform_doc(doc, output):
    org_str = ""
    for key in output:
        org_str += output[key] + " "
    doc = nlp(org_str)
    # clean person entities
    doc = clean_person(doc)
    return doc


#   function to return the pointer
def get_pointer(index, _list):
    for i in range(len(list(_list.keys()))):
        if index == list(_list.keys())[i]:
            pointer = i
    return pointer


#   function to return the first from start
def find_head(index, _list, str_start):
    t = index
    # ignore the start point of conj
    while _list[t] == 'conj':
        # print(t)
        str_start += 1
        t = list(_list.keys())[str_start]
    return t


#   function to return the first from the end
def find_tail(index, _list, str_end):
    t = index
    # ignore the start point of conj
    while _list[t] == 'conj':
        # print(t)
        str_end -= 1
        t = list(_list.keys())[str_end]
    return t


#   function to clean person entites into understandable phrase
def clean_person(doc):
    '''
    convert "person" / "a person" or "people" / "a group of people" / "group of people" / "num people"
    into shortened phrase
    example:
    '''
    output = {}
    for token in doc:
        output[token.i] = token.text

    person_list = {}
    for token in doc:
        if token.text == ("person"):
            person_list[token.i] = 'person'
            if token.i != 0 and token.nbor(-1).lower_ in ('a', 'one'):
                person_list[token.i - 1] = 'det'
        if token.text in (",", "and", "with"):
            person_list[token.i] = 'conj'
        if token.text == ("people"):
            person_list[token.i] = 'people'
            if token.i != 0 and token.nbor(-1).like_num:
                person_list[token.i - 1] = 'num'
            t = token
            while t.i != 0 and t.nbor(-1).lower_ in ('group', 'of', 'groups', 'a'):
                t = t.nbor(-1)
                person_list[t.i] = 'det'
    # sort list by index
    person_list = collections.OrderedDict(sorted(person_list.items()))
    # print([[i, person_list[i]] for i in person_list])
    # count person into group with num of person
    # group index into group
    entities_group = []
    person_group_str = {"start": 0, "end": 0}
    last_index = -1

    for i in person_list:
        # print(i,last_index, )
        if last_index != -1:
            if i - last_index > 1:
                person_group_str["end"] = last_index
                entities_group.append(person_group_str)
                person_group_str = {"start": i, "end": 0}
        elif last_index == -1:
            person_group_str["start"] = i
        last_index = i
        if last_index == list(person_list.keys())[-1]:
            person_group_str["end"] = last_index
            entities_group.append(person_group_str)
    # print("Grouping person enties :")
    # print()
    for item in entities_group:
        people_cnt = 0
        for i in range(item["start"], item["end"] + 1):
            if person_list[i] == "person":
                people_cnt += 1
            elif person_list[i] == "people":
                if i != 0 and doc[i].nbor(-1).like_num and doc[i].nbor(-1).lower_ in ('2', 'two'):
                    people_cnt += 2
                else:
                    people_cnt += 3
        if people_cnt == 1:
            # print(item,"a person")
            # get the key index of given index from the list
            str_start = get_pointer(item["start"], person_list)
            # ignore the start point of conj
            head = find_head(item["start"], person_list, str_start)
            output[head] = "a person"
            # ignore the conj at the end
            str_end = get_pointer(item["end"], person_list)
            # find the tail
            tail = find_tail(item["end"], person_list, str_end)
            # print(t)
            for i in range(head + 1, tail + 1):
                del output[i]
        elif people_cnt == 2:
            # print(item,"two people")
            # get the key index of given index from the list
            str_start = get_pointer(item["start"], person_list)
            # ignore the start point of conj
            head = find_head(item["start"], person_list, str_start)
            output[head] = "two people"
            # ignore the conj at the end
            str_end = get_pointer(item["end"], person_list)
            # print(t)
            # find the tail
            tail = find_tail(item["end"], person_list, str_end)
            # print(t)
            for i in range(head + 1, tail + 1):
                del output[i]
        elif people_cnt >= 3:
            # print(item,"a group of people")
            # get the key index of given index from the list
            str_start = get_pointer(item["start"], person_list)
            # ignore the start point of conj
            head = find_head(item["start"], person_list, str_start)
            output[head] = "a group of people"
            # ignore the conj at the end
            str_end = get_pointer(item["end"], person_list)
            # find the tail
            tail = find_tail(item["end"], person_list, str_end)
            # print(head,tail)
            for i in range(head + 1, tail + 1):
                del output[i]

        # print(item,people_cnt)

    # print([output[k] for k in output])
    org_str = ""
    for key in output:
        org_str += output[key] + " "
    doc = nlp(org_str)
    return doc


#   function to get the valid sentence for each paragragh
def sentence_filter(org_str):
    doc = nlp(org_str)
    # get the first sent of the doc
    doc = next(iter(doc.sents or []), None)
    # if sentence doesn't make sense, then use the second one to avoid index
    # TO DO
    return doc


#  function for data_preparation
def process(org_str):
    '''
    :param org_str:
    :return: result string, total number of proper noun entities found in the string, total entities identified in the string
    '''
    total_ents, total_noun_chunks = get_report(org_str)
    doc = sentence_filter(org_str)

    output = {}

    for token in doc:
        output[token.i] = token.text

    #  indexes list that need to remove from sentence
    indexes = []
    #  parent nodes of the named entities
    parent_node_list = {}
    #  named entities list
    entities_list = []
    #  person_list
    people_list = []
    #  chunks modify list
    chunk_list = []
    #  list for person, chunk and reform
    update_list = {}

    # print("Step 1: PRONOUN cleaning")
    for ent in doc.ents:
        # print(ent.text, ent.label_, ent.root.head.text)
        # print(ent.text, ent.label_)
        # todo: deal with LOC and WORK_OF_ART
        # For non person nor Carinal type, check their parent nodes, if their parent node is null, then abort;
        # if parent node is not null, then remove parent node along with all children associated
        # if ent.label_ in ('DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'LANGUAGE', 'LAW'):
        # delete
        if (ent.label_ != 'PERSON' and ent.label_ != 'CARDINAL'):
            # print("\""+ent.text+"\"", "will be removed along with its parent node:","\""+ent.root.head.text+"\"")
            # print([[node.text, node.pos_] for node in ent.subtree])
            for node in ent.subtree:
                if node.i not in indexes:
                    indexes.append(node.i)
                    # print(node.text, 'will be removed')
            parent_node_list[ent.root.head.i] = ent.root.head
            entities_list.append(ent.root.head.i)
            entities_list.append(ent.root.i)
        elif ent.label_ == 'PERSON':
            # print("person entity identified:",ent.text)
            # replace the span field with a single token of 'person'
            people_list.append(ent)

    replace_person_name(people_list, update_list, indexes)
    # print()

    # print(len(people_list), " person found, will be replace with person")
    # print("Step 2: Parent nodes cleaning")

    for key in parent_node_list:
        # print(parent_node_list[key], [[child.text, child.pos_] for child in parent_node_list[key].children])
        # print(parent_node_list[key], [[child.text, child.pos_] for child in parent_node_list[key].lefts], [[child.text, child.pos_] for child in parent_node_list[key].subtree])
        # print(parent_node_list[key].i, parent_node_list[key].text, "is noun")
        loop_tree(parent_node_list[key], indexes)

    # print()
    # print("Step 3: NOUN entities cleaning")
    for chunk in doc.noun_chunks:
        # print(chunk.text, chunk.label_)
        noun_exclude_list = entities_list[:]
        for person in people_list:
            for i in range(person.start, person.end):
                noun_exclude_list.append(i)
        # replace noun phrase with root text
        if chunk.root.i not in noun_exclude_list:
            chunk_list.append(chunk)
            # print("\""+chunk.text+"\"", 'will be replace with', "\""+chunk.root.text+"\"", [[child.text,child.pos_] for child in chunk.root.children])

    replace_chunk_name(chunk_list, indexes)

    # for token in doc:
    #  print("{5}: {0}/{1} <--{2}-- {3}/{4} --> ".format(token.text, token.pos_, token.dep_, token.head.text, token.head.pos_, token.i), [[child.pos_,child.text] for child in token.children])
    # print()

    # print original str
    # print("Original String: ")
    # print(org_str)

    # doc = remove_span(doc, indexes)
    # update person entities with "person"

    for key in update_list:
        output[key] = update_list[key]
    # remove tokens with the index in the indexes from the sentence
    for i in indexes:
        del output[i]
    # print("Step 4: Reform")
    # print()

    # print("Format with understandable phrases: ")
    # print()
    # reform the sentence
    doc = reform_doc(doc, output)

    # print final output
    # print()
    # print("Final output: ")
    # print(doc.text)

    return doc.text, total_ents, total_noun_chunks


#   function to get total entities and noun chunks
def get_report(org_str):
    doc = sentence_filter(org_str)
    #  total entities
    total_ents = len(list(doc.ents))
    #  total noun chunks
    total_noun_chunks = len(list(doc.noun_chunks))
    return total_ents, total_noun_chunks


#   function to get sentences, entities, noun chunks from the str, useful when checking if still ents left after nlp
def get_ents(org_str):
    '''
    :param org_str:
    :return: dict object containing sentences with entities, noun chunks and tokens
    '''
    doc_ents = {'sentences': [], 'ents': []}

    for sent in nlp(org_str).sents:
        doc_ents['sentences'].append(sent.text)

    for ent in nlp(org_str).ents:
        if ent.label_!= 'CARDINAL':
            doc_ents['ents'].append({'text': ent.text, 'type': ent.label_})
    '''
    print(ent.text, ent.label_, ent.root.head, [child for child in ent.root.head.lefts])
    if ent.label_ == 'PERSON':
        print(ent.root.text, [child.text for child in ent.root.lefts])
  

    for noun in nlp(org_str).noun_chunks:
        print(noun.text, noun.label_)

    for token in nlp(org_str):
        print(token.text, [child.text for child in token.subtree], token.head.text)
    '''
    return doc_ents

'''
#Example List
#Ute couple with child stand in front of old cars in the 1930\'s.
#couple with child stand in front of cars
#PASSED

#Two unidentified Ute Girls near the Whiterocks School.
#Two girls.
#PASSED

#I just bought two shares at 9 a.m. because the stock went up 30% in just 2 days according to the WSJ.
#I just bought two shares.
#PASSED


#A woman tends to a fire at a Ute Indian camp near Brush Creek.
#A woman tends to a fire at a camp
#PASSED


#Sidney and Eugenia Atwine with daughters
#FAILED, Sidney and Eugenia Atwine recognized as ORG
#Mike and Eugenia Atwine with daughters replaced with person name instead


#Nelson G. Sowards and other hay crew take a break from thier work of hauling hay at the Gibson/Soward place. 
#person and hay crew take a break from work of hauling hay at the place.
#PASSED

#Nelson G. Sowards and other hay crew take a break from thier work of hauling hay at the Gibson/Soward place. This farm was first homesteaded and built by Billy Gibson. Later his daughter and son-in-law, Mary and Nelson G. Sowards, lived on and continued the farm and ranch. Nelson's son Leland and Ruth Sowards later owned the property and it is still owned by the Soward family. The farm is located at 3110 North 250 West and is listed on the Uintah County Landmark Register.

#Nelson G. Sowards and other hay men haul hay at the Gibson/Soward place. 
#person and hay men haul hay at the place.
#PASSED

#With the steady growth of the Glines area it became necessary for a new chapel
#With the growth of the area it became necessary for a chapel
#PASSED

#A womens group is photographed at the Ashley Power Plant up Taylor Mountain road
#A womens group is photographed
#PASSED

#A large woman group, 1st great school president ,Mike Johnson, Nelson G. Sowards and two men are photographed at the Ashley Power Plant up Taylor Mountain road
#A woman group, school president, person, person and two men are photographed
#PASSED

#A women group is photographed at the Ashley Power Plant up Taylor Mountain road. Among the group are Vennie Lee, Mildred Hacking, Merle Oaks, Nora Long Rasmussen, Blanche Smith and Beaula Duvall McConkie.
#A woman group , school president , two people and two men are photographed
#PASSED

Challenge
#   Photo taken during the Allied occupation of Japan after World War II, 1949-1952; shows a Japanese woman cleaning bricks from the German Embassy for resale
#	1. Kathleen Sauter, three-year old daughter of Mr. and Mrs. Walter Sauter of Topanga, California, pauses in front of giant dinosaur bone at Dinosaur National Monument. They were among the thousands of visitors this summer.
#   Photo of Mrs. Louise Hansen Lyman, Delta Elementary School, Delta, Utah
#   Oral Watkins is pictured her senior year at Uintah High School where she was involved in many organizations. She is the daughter of William and Amanda Watkins. She worked for the J.C. Penney's store in Vernal and was transferred to the Salt Lake City store. She married Stanley Paxton on June 11, 1954. She later married Kenneth G. Anderson. She died March 11, 2008.
'''

# define the string to process here
# str = '	Leo Thorne began his long career in pbotography in 1906 when he began developing film. He purhased a studio in 1907 and never looked back. Not only did he photography people but he travele the valey taking photos of anything and everything. He was a charter member of the Lion\'s Club and a civic leader of the community.'
# new_str = process(str)
# print(get_report(str))