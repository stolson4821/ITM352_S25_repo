##Part A
responses = [5,7,3,8]
responses.append(0)
#responses.insert(2,6)
#print(responses)

##Part B
responses = responses[:2] + [6] + responses[2:]
print(responses)