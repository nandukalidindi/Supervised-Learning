# *****************************************************************
# NOTE: Please place the training and test CSV files in the same folder as the file
# NOTE: ruby kNN.rb to run the program
# *****************************************************************

class Array
  # Internal: Returns the highest frequent element in the given array
  #
  # Return: Element
  #
  # Examples
  #   [1, 2, 3, 1, 1, 1, 1].mode
  #   => 1
  def mode
    count_map = {}
    self.each { |x| count_map[x] = (count_map[x] || 0) + 1 }
    count_map.max_by{ |k, v| v }.first
  end

end

# Internal: Algorithm which returns K nearest similar neighbours
#
#
# Return: Array
#
# Examples
#   kNN(data, 3)
def kNN(input, k=3)
  # Store euclidian distance in the third index of the preprocessed data
  data.each { |x| x[2] = euclidian_measure(x[0], input) }

  # Sort on the euclidian distance i.e third index
  # And get the first K elements in the array
  data.sort_by { |s| s[2] }.first(k)
end

# Internal: Memoization method to prevent redundant data preprocess
#
# Return: Array
#
# Examples
#   data
def data
  # This is a common memoization technique used in Ruby
  @data ||= preprocess_data
end

# Internal: Read data from an input CSV file
#
# Return: Array
#
# Examples
#   preprocess_data("votes-train.csv")
#   => [[[features], 1], [[features], 2], [[features], 3]]
def preprocess_data(filename="votes-train.csv")
  lines = []
  File.open(filename, "r") do |f|
    f.each_line do |line|
      # Read another line if the next line is not empty
      # chomp will remove any trailing next line characters
      # Eg: "someone\n".chomp => "someone"
      next if line.chomp.empty?
      partition = line.partition(',') # Divide the string at classification value
      point = partition.last.chomp.split(',').map(&:to_f)
      classification = partition.first.to_f
      lines << [point, classification] # Arrange data for easy processing
    end
  end
  lines
end

# Internal: Euclidian distance calculator
#
# Return: Float
#
# Examples
#   euclidian_measure([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0])
#   => 2
def euclidian_measure(p1, p2)
  size = [p1.size, p2.size].max
  sum = (0..size-1).inject(0.0) { |sum, i| sum + ( ( (p1[i] || 0) - (p2[i] || 0) ) ** 2 ) }
  sum ** 0.5
end

# Test data for prediction efficiency
def predictor
  correct = 0
  wrong = 0
  File.open("votes-test.csv", "r") do |f|
    index = 0
    f.each_line do |line|
      next if line.chomp.empty?
      partition = line.partition(',')
      actual_value = partition.first
      input = partition.last.chomp.split(',').map(&:to_f)
      index += 1
      prediction = kNN(input).map { |x| x[1] }.mode
      if prediction == actual_value.to_f
        correct += 1
      else
        wrong += 1
      end
    end
  end

  p "CORRECT: #{correct}"
  p "WRONG: #{wrong}"
end

predictor
