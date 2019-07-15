from __future__ import unicode_literals
from library import str_nlp

test_str = raw_input("Please enter str that need to process: \n").decode('utf-8')
result_st, total_proper_nouns, total_entities = str_nlp.process(test_str)
print("Process Completed!")
print("")
print("Cleaned String: %s \nTotal Proper-Noun Identified: %d \nTotal Entities in the String: %d \n" % (result_st, total_proper_nouns, total_entities))