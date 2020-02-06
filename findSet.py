import numpy as np
import itertools
"""
For testing

card1 ={"id":"one_green_empty_diamond","color":"green","shape":"diamonds","fill":"empty","number":"one"}
card2 ={"id":"one_red_empty_diamond","color":"red","shape":"diamonds","fill":"empty","number":"one"}
card3 ={"id":"one_blue_empty_diamond","color":"blue","shape":"diamonds","fill":"empty","number":"one"}
card4 ={"id":"three_purple_full_diamonds","color":"purple","shape":"diamonds","fill":"full","number":"three"}
card5 ={"id":"three_green_full_diamonds","color":"green","shape":"diamonds","fill":"full","number":"three"}
card6 ={"id":"three_red_full_diamonds","color":"red","shape":"diamonds","fill":"full","number":"three"}
collection = [card1,card2, card3,card4,card5,card6]
"""
def isSet(triplet):
    "Return true if the triplet is a set"
    result =False
    if(triplet[0].get('color')==triplet[1].get('color')==triplet[2].get('color') or triplet[0].get('color')!=triplet[1].get('color')!=triplet[2].get('color')) :
        if(triplet[0].get('shape')==triplet[1].get('shape')==triplet[2].get('shape') or triplet[0].get('shape')!=triplet[1].get('shape')!=triplet[2].get('shape')) :
            if(triplet[0].get('fill')==triplet[1].get('fill')==triplet[2].get('fill') or triplet[0].get('fill')!=triplet[1].get('fill')!=triplet[2].get('fill')) :
                if(triplet[0].get('number')==triplet[1].get('number')==triplet[2].get('number') or triplet[0].get('number')!=triplet[1].get('number')!=triplet[2].get('number')) :
                    result= True
    return result

def getTriplets(collection):
    "Return all the combination of 3 elements in the collection"
    return itertools.combinations(collection, 3)

def getCardNames(triplet):
    "Return the card name of the triplet"
    names = []
    for card in triplet:
        names.append(card.get('id'))
    return names

def getSets(collection) :
    "Return all the sets"
    sets = []
    for triplet in getTriplets(collection):
        if isSet(triplet):
            sets.append(getCardNames(triplet))
    return sets


    