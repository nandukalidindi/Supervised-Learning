# *****************************************************************
# NOTE: Please place the training and test CSV files in the same folder as the file
# NOTE: ruby perceptron.rb to run the program
# *****************************************************************

@m = 0.1
@iterations = 1000
@bias = rand(0.0..1.0)
@theta = 0
@feature_count = 9

# Internal: Iteratively correct the separating multi dimensional place to divide
#           the data into meaningful parts
#
# Return: Array (Weight Vector)
#
# Examples
#   perceptron
def perceptron
  @feature_count = data.first.first.length
  iterations = 0
  weights = @feature_count.times.map { |x| 0.0001 }

  while iterations != @iterations
    data.each do |entry|
      feature_vector = entry.first
      activation = activation_output(weights, feature_vector)
      error = entry.last - activation
      error = (error == 0) ? 0.0001 : error
      @bias += @m*error
      weights = vector_add(weights, scalarXvector(error*@m, feature_vector))
    end
    iterations += 1
  end
  weights
end

# Internal: Multiply a vector with a scalar
#
# Return: Array (vector)
#
# Examples
#   activation_output([1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
#   => 0
def activation_output(w, x)
  sum = (0..@feature_count-1).inject(0.0) { |sum, i| (sum + (w[i] * x[i])) }
  ((sum + @bias) >= @theta) ? 1 : 0
end

# Internal: Multiply a vector with a scalar
#
# Return: Array (vector)
#
# Examples
#   scalarXvector(10.0, [1.0, 2.0, 3.0])
#   => [10.0, 20.0, 30.0]
def scalarXvector(scalar, array)
  array.map { |x| scalar * x }
end

# Internal: Memoization method to prevent redundant data preprocess
#
# Return: Array (vector)
#
# Examples
#   vector_add([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
#   => [2.0, 4.0, 6.0]
def vector_add(w, x)
  sum_vector = @feature_count.times.map { |x| 0 }
  (0..@feature_count-1).each { |i| sum_vector[i] = w[i] + x[i] }
  sum_vector
end

# Internal: Memoization method to prevent redundant data preprocess
#
# Return: Array
#
# Examples
#   data
def data
  # This is a common memoization technique used in Ruby
  @data ||= normalize_data
end

# Internal: Read data from an input CSV file
#
# Return: Array
#
# Examples
#   preprocess_data("votes-train.csv")
#   => [[[features], 1], [[features], 2], [[features], 3]]
def preprocess_data(file_name = "votes-train.csv")
  classified_data = []
  File.open(file_name, "r") do |f|
    f.each_line do |line|
      next if line.chomp.empty?
      partition = line.partition(",")
      classified_data << [partition.last.chomp.split(",").map(&:to_f), partition.first.to_f]
    end
  end
  classified_data
end

# Internal: Normalize data to get all values to [0, 1] range. Feature Scaling!
#
# Return: Array
#
# Examples
#   normalize_data("votes-train.csv")
#   => [[[features], 1], [[features], 2], [[features], 3]] Scaled data
def normalize_data(file_name = "votes-train.csv")
  data = preprocess_data(file_name)
  ranges = []
  (0..@feature_count-1).each do |i|
    feature_array = data.map { |x| x[0][i] }
    sum = feature_array.inject(0) { |sum, i| sum+i }/(feature_array.length)
    ranges[i] = [sum, feature_array.max - feature_array.min]
  end
  data.each do |x|
    x[0] = (0..@feature_count-1).map { |k| (x.first[k]-ranges[k][0])/(ranges[k][1]) }
  end
  data
end

# Test data for prediction efficiency
def predictor
  w = perceptron

  train_data = normalize_data("votes-test.csv")
  actual_values = train_data.map { |x| x.last }
  predictions = train_data.map { |x| activation_output(x.first, w) }

  wrong = (0..actual_values.length-1).select { |i| actual_values[i] != predictions[i] }.count
  correct = actual_values.length - wrong

  p "CORRECT: #{correct}"
  p "WRONG: #{wrong}"
end

predictor
