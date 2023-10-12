import Activation
import Layer
import Loss
import numpy as np 



def main():
    # 2 input nodes,
    # 1 hidden layer with 3 nodes,
    # 1 output node,
    #
    # creating the xor function
    # [1,0] means the result is one
    # [0,1] means the result is zero

    input = np.array([1,0])
    y_expect = np.array([1,0])

    hidden = Layer.Hidden(3, 2, 0.5)
    output = Layer.Output(2, 3, 0.5)

    # Hidden layer 
    hidden.forward(input)
    print("HIDDEN LAYER OUTPUT:\n", hidden.output, "\n\n")

    # Output Layer 
    output.forward(hidden.output)
    print("OUTPUT LAYER OUTPUT:\n", output.output, "\n\n")

    loss = Loss.SquaredError()
    loss.forward(y_expect, output.output)
    print("LOSS:\n", loss.output)

    # Backwards pass
    output_delta = output.backwards(y_expect, len(input))
    output.update(output.deltaWeights, output.deltaBias)

    hidden.backwards(output_delta, len(input))
    hidden.update(hidden.deltaWeights, hidden.deltaBias)





    
if __name__ == "__main__":
   main() 
