import matplotlib.pyplot as plt


def plot_forex_chart(data, special_color="red"):
    plt.figure(figsize=(10, 6))
    outcome = data.pop("outcome")

    plt.plot(list(data.values()), marker="s", markersize=8, linestyle="-", color="blue")  # Default color and square marker for regular nodes

    plt.title("Traider")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.xticks(range(len(data) - 1))  # Set x-axis ticks with spacing of 1

    # Find highest and lowest points
    highest_point = max(list(data.values()))
    lowest_point = min(list(data.values()))

    total_range = highest_point - lowest_point

    values_list = list(data.values())
    last_value = values_list[-1]
    print(last_value)

    val = last_value + (outcome * total_range)

    plt.plot((25), val, marker="s", markersize=10, color=special_color)  # Special color and square marker for specific indices
    plt.show()

    return


