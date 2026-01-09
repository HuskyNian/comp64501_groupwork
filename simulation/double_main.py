from DoubleSlot import QuantumChannelAnalyzer as q
import numpy as np

def main():
    # Example usage
    
    # Create an analyzer instance
    analyzer = q(g=(.8, 1., 1.1), r=0.3, t=5.0)
    
    # Create a shallow pocket model
    choi_superchannel = analyzer.create_shallow_pocket_model()

    # Create a parameterized unitary
    V1 = analyzer.parameterised_unitary(np.pi, np.pi/7, np.pi) # [0,2pi]
    print("\nParameterized unitary:")
    print(analyzer.qobj_round(V1, 2))

    V2 = analyzer.parameterised_unitary(np.pi, np.pi, 3*np.pi/4) # [0,2pi]
    print("\nParameterized unitary:")
    print(analyzer.qobj_round(V2, 2))
    
    # Create Choi state of the unitary
    choi_input1 = analyzer.choi_state_unitary(V1)
    print("\nInput channel1:")
    print(analyzer.qobj_round(choi_input1, 2))

    choi_input2 = analyzer.choi_state_unitary(V2)
    print("\nInput channel2:")
    print(analyzer.qobj_round(choi_input2, 2))

    # Obtain the Choi state of the output
    choi_output = analyzer.link_product(choi_superchannel, choi_input1, choi_input2)
    print("\nOutput channel:")
    print(analyzer.qobj_round(choi_output, 2))
    
    # Find the closest unitary and the corresponding fidelity (target)
    U, F_U = analyzer.closest_unitary_channel(choi_output)
    print("\nClosest unitary U (rounded):")
    print(analyzer.qobj_round(U, 2))
    print(f"Fidelity (Closest Unitary vs Output): {F_U:.2f}")

if __name__ == "__main__":
    main()