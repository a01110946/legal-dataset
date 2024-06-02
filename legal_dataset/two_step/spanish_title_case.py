# spanish_title_case.py

def spanish_title_case(title):
    # List of words that should not be capitalized unless they are the first word
    non_cap_words = ['de', 'del', 'y', 'e', 'las', 'la', 'los', 'para', 'por', 'con', 'sin', 'sobre', 'tras', 'el', 'en', 'a']
    
    # Split the title into words
    words = title.split()
    
    # Capitalize the first word
    words[0] = words[0].capitalize()
    
    # Iterate over the remaining words and capitalize them if they are not in the non_cap_words list
    for i in range(1, len(words)):
        if words[i].lower() not in non_cap_words:
            words[i] = words[i].capitalize()
        else:
            words[i] = words[i].lower()
    
    # Join the words back into a title
    return ' '.join(words)