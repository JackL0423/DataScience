from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# create line chart, years on x-axis, gdp on y-axis
#plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# add title
#plt.title("Nominal GDP")

# add a label to the y-axis
#plt.ylabel("Billions of $")

#plt.show()


# Bar charts

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Ghandi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# plot bars with left x-coordinates [0, 1, 2, 3, 4], heights [num_oscars]
#plt.bar(range(len(movies)), num_oscars)

#plt.title("My Favorite Movies")         # add a title
#plt.ylabel("# of Academy Awards")       # label the y-axis
#plt.xticks(range(len(movies)), movies)  # label x-axis with movie names at bar centers

#plt.show()


from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# Bucket grades by decile, but put 100 in with the 90s
#histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

#plt.bar([x + 5 for x in histogram.keys()],  # shift bars right by 5
#        histogram.values(),                 # give each bar its correct height
#        10,                                 # give each bar a width of 10
#        edgecolor=(0,0,0))                  # black edges for each bar

#plt.axis([-5, 105, 0, 5])                   # x-axis from -5 to 105
                                             # y-axis from 0 to 5

#plt.xticks([10 * i for i in range(11)])     # x-axis labels at 0, 10, ..., 100
#plt.xlabel("Decile")
#plt.ylabel("# of Students")
#plt.title("Distribution of Exam 1 Grades")
#plt.show()


# Line charts
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# We can make multiple cals to plt.plot
# to show multiple series on the same chart
#plt.plot(xs, variance,      'g-', label='variance')     # green solid line 
#plt.plot(xs, bias_squared,  'r-', label='bias^2')       # red dot-dashed line
#plt.plot(xs, total_error,   'b:', label='total error')  # blue dotted line

# Because we've assigned labels to each series,
# we can get a legend for free (loc=9 means "top center")
#plt.legend(loc=9)
#plt.xlabel("model complexity")
#plt.xticks([])
#plt.title("The Bias-Variance Tradeoff")
#plt.show()


# Scatterplots
friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

#plt.scatter(friends, minutes)

# label each point
#for label, friend_count, minute_count in zip(labels, friends, minutes):
#    plt.annotate(label,
#                 xy=(friend_count, minute_count),   # put the label with its point
#                 xytext=(5, -5),                    # but slightly offset
#                 textcoords='offset points')
#plt.title("Daily Minutes vs. Number of Friends")
#plt.xlabel("# of friends")
#plt.ylabel("daily minutes spent on the site")
#plt.show()
