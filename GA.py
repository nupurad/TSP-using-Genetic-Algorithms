import numpy as np
import random
import os

#Input function
def read_input():
    city_coordinates = []
    with open('input.txt', 'r') as file:
        num_cities = int(file.readline().strip())
        for _ in range(num_cities):
            line = file.readline().strip()
            coordinates = list(map(int, line.split()))
            city_coordinates.append(coordinates)
    return num_cities, city_coordinates

#Create population function
def create_population(pop_size, num_cities):
    initial_population = []
    for _ in range(pop_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        initial_population.append(individual)
    return initial_population

#Fitness function
def fitness(initial_population, city_coordinates):
    fitness_scores = []
    for individual in initial_population:
        total_distance = 0
        num_cities = len(individual)
        for i in range(num_cities):
            x1, y1, z1 = city_coordinates[individual[i]]
            x2, y2, z2 = city_coordinates[individual[(i + 1) % num_cities]]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            total_distance += distance
        fitness_scores.append((total_distance, individual))
    return fitness_scores

#Selection function 
def roulette_wheel_selection(fitness_scores, initial_population):
    total_fitness = sum(score[0] for score in fitness_scores)  
    fit_prob = [score[0] / total_fitness for score in fitness_scores]
    
    cumulative_prob = []
    cumulative_sum = 0
    for prob in fit_prob:
        cumulative_sum += prob
        cumulative_prob.append(cumulative_sum)

    selected_parents = []
    for _ in range(2):  
        rand_num = np.random.rand() 
        for i, cumulative in enumerate(cumulative_prob):
            if rand_num <= cumulative:
                selected_parents.append(initial_population[i])
                break

    parent1, parent2 = selected_parents
    return parent1, parent2

#Crossover function
def sequencial_constructive_crossover(parent1, parent2, city_coordinates):
    offspring = []
    city1 = parent1[0]
    offspring.append(city1)

    remaining_cities = set(parent1 + parent2)
    remaining_cities.remove(city1)

    while remaining_cities:
        last_city = offspring[-1]
        x1, y1, z1 = city_coordinates[last_city]
        
        min_distance = float('inf')
        next_city = None

        for city in remaining_cities:
            x2, y2, z2 = city_coordinates[city]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                next_city = city

        offspring.append(next_city)
        remaining_cities.remove(next_city)

    offspring.append(offspring[0])  
    return offspring

#Mutation function
def swap_mutation(offspring):
    mutated_offspring = offspring.copy()
    idx1, idx2 = random.sample(range(1, len(mutated_offspring) - 1), 2)

    mutated_offspring[idx1], mutated_offspring[idx2] = mutated_offspring[idx2], mutated_offspring[idx1]
    
    return mutated_offspring

#Genetic algorithm
def genetic_algorithm(city_coordinates, num_cities, pop_size=100, generations=500):
    population = create_population(pop_size, num_cities)
    
    for generation in range(generations):
        fitness_scores = fitness(population, city_coordinates)
        fitness_scores.sort(key=lambda x: x[0])
        parent1, parent2 = roulette_wheel_selection(fitness_scores, population)
        offspring = sequencial_constructive_crossover(parent1, parent2, city_coordinates)
        mutated_offspring = swap_mutation(offspring)
        population.append(mutated_offspring)  
        population.sort(key=lambda x: fitness([x], city_coordinates)[0][0]) 
        population = population[:pop_size]  
   
    best_solution = min(population, key=lambda x: fitness([x], city_coordinates)[0][0])
    best_distance = fitness([best_solution], city_coordinates)[0][0]

    best_path = best_solution.copy()  
    if best_path[-1] != best_path[0]:  
        best_path.append(best_path[0])  

    return best_distance, best_path

#Output function
def write_output(best_distance, best_path, city_coordinates, filename='output.txt'):
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, 'w') as file:
        file.write(f"{best_distance:.3f}\n")
        
        for city_index in best_path:
            coordinates = city_coordinates[city_index]
            file.write(f"{coordinates[0]} {coordinates[1]} {coordinates[2]}\n")

#Main function
def main():
    num_cities, city_coordinates = read_input()
    best_distance, best_path = genetic_algorithm(city_coordinates, num_cities)
    write_output(best_distance, best_path, city_coordinates)

if __name__ == "__main__":
    main()
