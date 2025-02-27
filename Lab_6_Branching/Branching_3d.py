
##DO THIS ON YOUR OWN

def determine_progress3(hits, spins):
    if spins == 0:
        return "Get going!"
    
    hits_spins_ratio = hits / spins
    
    if hits_spins_ratio >= 0.5 and hits < spins:
        return "You win!"
    elif hits_spins_ratio >= 0.25:
        return "Almost there!"
    elif hits_spins_ratio > 0:
        return "On your way!"
    else:
        return "Get going!"


##Starting point
def test_determine_progress(progress_function):
   # Test case 1: spins = 0 returns “Get going!”
    assert progress_function(10, 0) == "Get going!", "Test case 1 Passed"
    assert progress_function(1, 1) == "Almost there", "Test Case 2 Passed"
    assert progress_function(1, 5) == "On your way!", "Test Case 3 Passed"
    assert progress_function(5, 10) == "You win!", "Test Case 4 Passed"


print(determine_progress3(0,5))# Get Going! 
print(determine_progress3(1,1))#Almost there!
print(determine_progress3(1,5))#On your way!
print(determine_progress3(5,10))#You win!