# *****************************************************************
# NOTE: Please place the training and test CSV files in the same folder as the file
# NOTE: ruby ID3.rb to run the program
# *****************************************************************

@bins = 45
@features = 9

class Tree
  attr_accessor :root

  def initialize(root)
    @root = root
  end
end

class Node
  attr_accessor :index, :value, :children, :class_name

  def initialize(index, value, children, class_name)
    @index, @value, @children, @class_name = index, value, children, class_name
  end
end

# Internal: Tree construction by recursive pruning of data based on the maximum
#           information gain index
#           RECURSIVE Strategy
#
# Return: Node
#
# Examples
#   ID3(data, Node.new(0, 0, [], 1), Node.new(6, "Hispanic", nil, nil))
#   => Node.new(6, "Hispanic", nil, nil)
def ID3(data, node, root)
  index = get_highest_information_gain(data)
  children = data.map { |x| x.first[index] }.uniq.map { |y| Node.new(index, y, nil, nil)}
  if node.nil?
    node = Node.new(index, index, children, nil)
    root = node
  else
    node.children = children
  end
  node.children.each do |child|
    child_wise_values = data.select { |x| x.first[index] == (child.value) }
    class_values = child_wise_values.map{ |x| x.last }.uniq
    if class_values.count == 1
      child.children = nil
      child.class_name = class_values.first
    else
      ID3(child_wise_values, child, root)
    end
  end
  root
end

# Internal: Returns entropy of the classification
#
# Return: Float
#
# Examples
#   class_entropy(data)
#   => -1.13
def class_entropy(data)
  impurity = 0
  class_values = data.map { |x| x.last }.uniq
  class_values.each do |value|
    numerator = data.select { |x| x.last == value }.size
    probability = numerator.to_f/(data.size.to_f)
    impurity -= (probability.to_f * Math.log2(probability))
  end
  impurity
end

# Internal: Returns the entropy of a particular feature
#
# Return: Float
#
# Examples
#   feature_entropy(data, 3)
#   => 1.52
def feature_entropy(data, feature_index)
  impurity = 0
  feature_values = data.map { |x| x.first[feature_index] }.uniq
  feature_values.each do |feature_value|
    feature_set = data.select { |x| (x.first[feature_index]) == (feature_value) }
    class_values = feature_set.map { |x| x.last }.uniq
    class_values.each do |class_value|
      numerator = feature_set.select { |x| x.last == class_value }.size
      probability = numerator.to_f/(data.size.to_f)
      impurity += (probability.to_f * class_entropy(feature_set))
    end
  end
  impurity
end

# Internal: Returns the feature index which is responsible for highest
#           information gain
#
# Return: Integer
#
# Examples
#   get_highest_information_gain(data)
#   => 4
def get_highest_information_gain(data)
  feature_index = 0
  min_gain = -Float::INFINITY
  (0..@features-1).each_with_index do |x|
    gain = class_entropy(data) - feature_entropy(data, x)
    if gain > min_gain
      min_gain = gain
      feature_index = x
    end
  end
  feature_index
end

# Internal: Memoization method to prevent redundant data preprocess
#
# Return: Array
#
# Examples
#   data
def data
  # This is a common memoization technique used in Ruby
  @data ||= bin_data
end

# Internal: Read data from an input CSV file
#
# Return: Array
#
# Examples
#   preprocess_data("votes-train.csv")
#   => [[[features], 1], [[features], 2], [[features], 3]]
def preprocess_data(file_name="votes-train.csv")
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

# Internal: Categorize data based on the values for convert continuous features
#           to discrete. Hashing is used as a binning strategy and the bucket size
#           is decided by the @bins instance variable
#
# Feature 3 and 4 are treated differently by scaling to 100 and then hashed
# because of many values being the range of [0, 1]
#
# Return: Array
#
# Examples
#   bin_data
def bin_data(file_name="votes-train.csv")
  data = preprocess_data(file_name)
  data.each_with_index do |x, index|
    (0..@features-1).each do |k|
      if k == 3 || k == 4
        data[index].first[k] = ((data[index].first[k]*40).to_f).floor%(@bins)
      else
        data[index].first[k] = (data[index].first[k].to_f).floor%(@bins)
      end
    end
  end
  data
end

# Internal: Traverses the tree based on the input and makes a valid prediction
#           or returns nil if the traversal for input is not present in the tree
#
#
# Return: String / Float
#
# Examples
#   return_class(root.children, data)
def return_class(node, data)
  if node == nil
    result = node.class_name
  end
  node.each do |child|
    if (child.value) == (data[child.index])
      if child.children.nil?
        result = child.class_name
      else
        result = return_class(child.children, data)
      end
    end
  end
  result
end

def predictor
  tree = ID3(bin_data("votes-train.csv"), nil, nil)
  test_data = bin_data("votes-test.csv")
  nil_count = 0
  correct = 0
  test_data.each_with_index do |x, index|
    k = return_class(tree.children, x.first)
    if k.nil?
      nil_count += 1
    else
      correct += 1 if x.last == k
    end
  end

  p "CORRECT: #{correct}"
  p "WRONG: #{test_data.length - nil_count - correct}"
  p "Unable to predict: #{nil_count}"
end

predictor
