import tables
import math

type
  node_types = enum
    tanh_hidden, logist_hidden, input, output, bias
  node* = object of RootObj
    output: float
    sum: float
    weights: seq[float]
    neighbors: seq[int]
    error: float
    node_type: node_types
    weight_deltas: seq[float]
type
  network* = object of RootObj
    nodes: seq[node]
    input_layer: seq[int]
    last_layer: seq[int]
    bias_id: int
    learning_rate: float
method eval(x: var node): float =
  case x.node_type
  of logist_hidden:
    # x.output = math.tanh(x.sum)
    x.output = 1/(1+math.pow(math.E, -x.sum))
  of tanh_hidden:
    # x.output =
    x.output = math.tanh(x.sum)
  of bias:
    x.output = 1.0
  else:
    x.output = x.sum
  return x.output

method eval_prime(x: var node): float =
  case x.node_type:
  of logist_hidden:
    x.output = - x.output*(1-x.output)
  of tanh_hidden:
    x.output = (1 - x.output * x.output)
  else:
    return x.output
  return x.output

method set_inputs(net: var network, inputs: seq[float]) {.inline.} =
  for i, val in inputs:
    net.nodes[net.input_layer[i]].sum = val

method compute_values(net: var network) {.inline.} =
  for i, node_i in net.nodes:
    discard net.nodes[i].eval()
    for j, node_j_id in node_i.neighbors:
      var weight_i_j = node_i.weights[j]
      net.nodes[node_j_id].sum += net.nodes[i].output * weight_i_j

method wipe_clean(net: var network) {.inline.} =
  for i,node_i in net.nodes:
    net.nodes[i].sum = 0
    net.nodes[i].output = 0
    net.nodes[i].error = 0

method set_errors(net: var network, outputs: seq[float]) {.inline.} =
  for i, val in outputs:
    net.nodes[net.nodes.len - outputs.len + i].error = val - net.nodes[net.nodes.len - outputs.len + i].output

method compute_errors(net: var network){.inline.} =
  var j: int
  for index_from_top in 1..net.nodes.len:
    j = net.nodes.len - index_from_top
    if net.nodes[j].node_type in [tanh_hidden, logist_hidden]:
      # outputs already have errors and inputs ndon't need theres
      var post_sum: float = 0
      for i, node_i_id in net.nodes[j].neighbors:
        var weight_i_j = net.nodes[j].weights[i]
        var node_i = net.nodes[node_i_id]
        post_sum += node_i.error * weight_i_j
      net.nodes[j].error = net.nodes[j].eval_prime() * post_sum

method get_errors(net: var network, outputs: seq[float]){.inline.}=
  net.set_errors(outputs)
  net.compute_errors()

method compute_weight_deltas(net: var network) {.inline.} =
  for j, node_j in net.nodes:
    net.nodes[j].weight_deltas = @[]
    for i, node_i_id in node_j.neighbors:
      net.nodes[j].weight_deltas.add( node_j.output * net.nodes[node_i_id].error)

method calculate(net: var network, inputs: seq[float]): seq[float] =
  net.wipe_clean()
  net.set_inputs(inputs)
  net.compute_values()
  var output_fins: seq[float] = @[]
  for i in 0..(net.last_layer.len - 1):
    output_fins.add(net.nodes[net.nodes.len - net.last_layer.len + i].output)
  return output_fins

method update_weight_deltas(net: var network) {.inline.} =
  for j, node_j in net.nodes:
    for i, node_i_id in node_j.neighbors:
      net.nodes[j].weights[i]+= net.learning_rate * node_j.output * net.nodes[node_i_id].error

method update(net: var network, inputs: seq[float], outputs: seq[float]): seq[float] =
  # calculate wipes it clean for us
  var output_fins: seq[float] = net.calculate(inputs)
  net.get_errors(outputs)
  net.update_weight_deltas()
  # for i, node_i in net.nodes:
  #   for j, node_j_id in node_i.neighbors:
  #     net.nodes[i].weights[j] += net.learning_rate * net.nodes[i].weight_deltas[j] # 0.1 decreases rate
  return output_fins

method add_many_typed_layer(net: var network, node_types_to_add: seq[node_types]) {.inline.} =
  var next_layer: seq[int] = @[]
  for node_type in node_types_to_add:
    # node 0 is bias
    let new_node = node(node_type: node_type, neighbors: @[], weights: @[]);
    next_layer.add(net.nodes.len)
    net.nodes.add(new_node)
  for id in net.last_layer:
    for neighbor_id in next_layer:
      net.nodes[id].neighbors.add(neighbor_id)
      # 0.5 seems to be much faster
      net.nodes[id].weights.add(1.0 - math.random(2.0)) #this seems to be good for now
  net.last_layer = next_layer

method add_layer(net: var network, amount: int, node_type: node_types) {.inline.} =
  var types: seq[node_types] = @[]
  for i in 1..amount:
    types.add(node_type)
  net.add_many_typed_layer(types)

method add_output_layer(net: var network, amount:int) {.inline.} =
  net.add_layer(amount, output)

proc init_network(input_num: int, learning_rate: float): network =
  var net: network = network(nodes: @[], input_layer: @[], last_layer: @[])
  net.learning_rate = learning_rate
  net.nodes.add(node(node_type: bias, neighbors: @[], weights: @[])) #sets bias
  net.last_layer.add(0)
  net.bias_id = 0
  for i in 1..input_num:
    # node 0 is bias
    net.nodes.add(node(node_type: input, neighbors: @[], weights: @[]))
    net.input_layer.add(i)
    net.last_layer.add(i)
  return net
var nodes: network = init_network(3, 0.1)
nodes.add_layer(14, tanh_hidden)
nodes.add_layer(12, tanh_hidden)
nodes.add_layer(8, tanh_hidden)
nodes.add_layer(4, tanh_hidden)
nodes.add_output_layer(1)
# var bias1: node = node(neighbors: @[3, 4, 5], weights: @[0.1, 0.1, 0.1], node_type: bias)
# var input1: node = node(neighbors: @[3,4], weights: @[0.1, 0.1], node_type: input)
# var input2: node = node(neighbors: @[3,4], weights: @[0.1, 0.1], node_type: input)
# var hidden1: node = node( neighbors: @[5], weights: @[0.1], node_type: hidden)
# var hidden2: node = node( neighbors: @[5], weights: @[0.1], node_type: hidden)
# var output1: node = node(neighbors: @[], weights: @[], node_type: output)
# var nodes: network = network(nodes: @[bias1, input1, input2, hidden1, hidden2, output1], input_layer: @[1,2], last_layer: @[5])
# Problems in numbering and assigning inputs or outputs it seems
var x,y,z,w: float
var total = 0.0
var val : seq[float]
for i in 0..1000000:
  x = float(math.random(1.0))
  y = float(math.random(1.0))
  z = float(math.random(1.0))

  w = (x*y+x*z+(y*z*3)+x*x*y+2-3*z*x)/100.0
  val =nodes.update(@[x,y,z],@[w])
  total+= abs(val[0]-w)
  if i mod 5000 == 0 and i!=0:
    echo total/ 5000
    if total/5000.0 < 1.0e-10:
      for i,x in nodes.nodes:
        echo i,": ",x.neighbors, " ", x.weights
      break
    total = 0.0
  # if i mod 1000000 == 0 and i!=0:
  #   for i,x in nodes.nodes:
  #     echo i,": ",x.neighbors, " ", x.weights



#echo calculate(nodes,  @[1.0,1.0])[0] - 0.0
# nim compile --run multilayer.nim
# if you want super speed after compiling:
# nim compile --run --opt=speed multilayer.nim
