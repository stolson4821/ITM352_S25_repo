def determine_progress1(hits, spins):
    if spins == 0:
        return "Get going!"
    
    hits_spins_ratio = hits / spins

    if hits_spins_ratio > 0:
        progress = "On your way!"
        if hits_spins_ratio >= 0.25:
            progress = "Almost there!"
            if hits_spins_ratio >= 0.5:
                if hits < spins:
                    progress = "You win!"
    else:
        progress = "Get going!"

    return progress

##Starting point
def test_determine_progress(progress_function):
   # Test case 1: spins = 0 returns “Get going!”
    assert progress_function(10, 0) == "Get going!", "Test case 1 Passed"
    assert progress_function(1, 1) == "Almost there", "Test Case 2 Passed"
    assert progress_function(1, 5) == "On your way!", "Test Case 3 Passed"
    assert progress_function(5, 10) == "You win!", "Test Case 4 Passed"

print(determine_progress1(0,5))# Get Going! 
print(determine_progress1(1,1))#Almost there!
print(determine_progress1(1,5))#On your way!
print(determine_progress1(5,10))#You win!
print(test_determine_progress)