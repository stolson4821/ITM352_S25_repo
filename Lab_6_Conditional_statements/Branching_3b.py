def determine_progress2(hits, spins):
    if spins == 0:
        return "Get going!"
    
    hits_spins_ratio = hits / spins
    
    if hits_spins_ratio >= 0.5 and hits < spins:
        return "You win!"
    
    if hits_spins_ratio >= 0.25:
        return "Almost there!"
    
    if hits_spins_ratio > 0:
        return "On your way!"
    
    return "Get going!"


##Starting point
def test_determine_progress(progress_function):
   # Test case 1: spins = 0 returns “Get going!”
    assert progress_function(10, 0) == "Get going!", "Test case 1 Passed"
    assert progress_function(1, 1) == "Almost there", "Test Case 2 Passed"
    assert progress_function(1, 5) == "On your way!", "Test Case 3 Passed"
    assert progress_function(5, 10) == "You win!", "Test Case 4 Passed"


print(determine_progress2(0,5))# Get Going! 
print(determine_progress2(1,1))#Almost there!
print(determine_progress2(1,5))#On your way!
print(determine_progress2(5,10))#You win!