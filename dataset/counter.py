#write a simple script that opens the txt file cites.txt and counts how many times each city appears in the file.
#The file contains one city per line.

from collections import Counter
def count_cites(file_path):
    with open(file_path, 'r') as file:
        cites = file.readlines()
    
    # Remove any whitespace characters like `\n` at the end of each line
    cites = [city.strip() for city in cites]
    
    # Use Counter to count occurrences of each city
    city_counts = Counter(cites)
    
    return city_counts
if __name__ == "__main__":
    file_path = 'cities.txt'
    city_counts = count_cites(file_path)
    for city, count in city_counts.items():
        print(f'{city} - {count}')
        
        