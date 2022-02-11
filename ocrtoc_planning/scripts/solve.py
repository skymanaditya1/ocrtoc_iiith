"""Simple Pickup Delivery Problem (PDP)."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    from scipy.spatial import distance_matrix

    #nodes = [[0, 0], [41, 44], [2, 1], [27, 5], [3, 49], [20, 4], [18, 4], [39, 44], [23, 12]] # 2431
    #nodes = [[0, 0], [23, 28], [34, 28], [42, 46], [11, 44], [19, 25], [36, 27], [0, 25], [7, 38]] # 4123
    #nodes = [[6, 6], [16, 27], [14, 2], [24, 45], [6, 14], [49, 27], [44, 0], [17, 11], [36, 17]] # 4132
    #nodes = [[0, 0], [43, 19], [18, 47], [41, 38], [50, 16], [45, 12], [32, 17], [4, 5], [43, 32]] # 2143
    #nodes = [[6, 6], [11, 47], [27, 4], [5, 48], [21, 34], [26, 19], [37, 10], [8, 37], [29, 9]] # 3142
    # nodes = [[0, 0], [31, 34], [19, 27], [9, 19], [12, 34], [43, 32], [5, 26], [17, 25], [46, 43]] # 2301 Working default
    #nodes = [[9, 1], [22, 29], [49, 35], [20, 19], [8, 13], [8, 13], [16, 18], [13, 11]] # 1432
    # Trying 3D nodes
    nodes = [[0, 0, 0], [31, 34, 31], [19, 27, 1], [9, 19, 5], [12, 34, 10], [43, 32, 38], [5, 26, 65], [17, 25, 20], [46, 43, 90] ]

    dm = distance_matrix(nodes, nodes)

    data['distance_matrix'] = dm
    data['pickups_deliveries'] = [
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8]
        # [0, 4],
        # [1, 5],
        # [2, 6],
        # [3, 7]
    ]
    data['num_vehicles'] = 1
    data['depot'] = 0
    
    data['demands'] = [0, 1, 1, 1, 1, -1, -1, -1, -1] # [1, 1, 1, 1, -1, -1, -1, -1]
    data['vehicle_capacities'] = [1]
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    import time
    t = time.time()
    total_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        total_distance += route_distance
    print("total time taken:", time.time()-t)
    print('Total Distance of all routes: {}m'.format(total_distance))


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    # Define cost of each arc.
    def distance_callback(from_index, to_index):
        """Returns the manhattan distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        2,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')


    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        10000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Define Transportation Requests.
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
    print(solution)


if __name__ == '__main__':
    main()