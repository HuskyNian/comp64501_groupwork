import numpy as np
from qutip import *

class QuantumChannelAnalyzer:
    """
    A class for analyzing quantum channels and superchannels in the shallow-pocket model.
    
    This class provides methods for:
    - Creating shallow-pocket model superchannels
    - Computing link products of superchannels and channels
    - Finding closest unitary approximations to quantum channels
    - Working with parameterized unitaries
    """
    
    def __init__(self, g=None, r=None, t=None):
        """
        Initialize the QuantumChannelAnalyzer.
        
        Parameters
        ----------
        g : tuple[float, float] or list[float, float]
            (g1, g2) coupling strengths for history and future segments
        r : float, optional
            Lorentzian width parameter (> 0)
        t : float, optional
            Evolution time for both segments
        """
        self.t = t
        self.r = r
        self.g1, self.g2 = g
    
    @staticmethod
    def qobj_round(qobj, digits=2):
        """
        Round the matrix elements of a Qobj to specified decimal places.
        
        Parameters
        ----------
        qobj : qutip.Qobj
            Quantum object to round
        digits : int
            Number of decimal places
            
        Returns
        -------
        qutip.Qobj
            Rounded quantum object
        """
        data = np.round(qobj.full(), digits)
        return Qobj(data, dims=qobj.dims)
    
    def create_shallow_pocket_model(self, g=None, r=None, t=None):
        """
        Construct the Choi state of the superchannel for the shallow-pocket model.
        
        Parameters
        ----------
        g : tuple[float, float] or list[float, float], optional
            (g1, g2) coupling strengths for history and future segments
            (uses instance values if not provided)
        r : float, optional
            Lorentzian width parameter (uses instance value if not provided)
        t : float, optional
            Evolution time (uses instance value if not provided)
            
        Returns
        -------
        qutip.Qobj
            Choi state representing the superchannel
        """
        # Use provided parameters or fall back to instance attributes
        t = t if t is not None else self.t
        r = r if r is not None else self.r
        g1, g2 = (self.g1, self.g2) if g is None else g

        # Validate required params AFTER fallback
        if any(x is None for x in (g1, g2, r, t)):
            raise ValueError("Need g=(g1,g2), r, t (either via init or arguments).")
        if r <= 0:
            raise ValueError("r must be > 0.")
        
        # Define basis states for 4-qubit system
        ket_0000 = basis(16, 0)   # |0000⟩
        ket_0011 = basis(16, 3)   # |0011⟩
        ket_1100 = basis(16, 12)  # |1100⟩
        ket_1111 = basis(16, 15)  # |1111⟩
        
        # Calculate exponential terms
        exp_neg_trg1 = np.exp(-t * r * g1)
        exp_neg_trg2 = np.exp(-t * r * g2)
        exp_neg_trg1plusg2 = np.exp(-t * r * (g1 + g2))
        exp_neg_trg1minusg2 = np.exp(-t * r * np.abs(g1 - g2))
        
        # Build operator using QuTiP's ket-bra notation
        I = (ket_0000 * ket_0000.dag() + 
             exp_neg_trg2 * ket_0011 * ket_0000.dag() + 
             exp_neg_trg1 * ket_1100 * ket_0000.dag() + 
             exp_neg_trg1plusg2 * ket_1111 * ket_0000.dag() +
             
             exp_neg_trg2 * ket_0000 * ket_0011.dag() + 
             ket_0011 * ket_0011.dag() + 
             exp_neg_trg1minusg2 * ket_1100 * ket_0011.dag() + 
             exp_neg_trg1 * ket_1111 * ket_0011.dag() +
             
             exp_neg_trg1 * ket_0000 * ket_1100.dag() + 
             exp_neg_trg1minusg2 * ket_0011 * ket_1100.dag() + 
             ket_1100 * ket_1100.dag() + 
             exp_neg_trg2 * ket_1111 * ket_1100.dag() +
             
             exp_neg_trg1plusg2 * ket_0000 * ket_1111.dag() + 
             exp_neg_trg1 * ket_0011 * ket_1111.dag() + 
             exp_neg_trg2 * ket_1100 * ket_1111.dag() + 
             ket_1111 * ket_1111.dag())
        
        I.dims = [[2, 2, 2, 2], [2, 2, 2, 2]]
        
        return I
    
    @staticmethod
    def link_product(superchannel, input_channel):
        """
        Compute the link product of a superchannel and input channel.
        
        The link product gives the output channel.
        Superchannel shape: H_3 ⊗ H_2 ⊗ H_1 ⊗ H_0
        Input channel shape: H_2 ⊗ H_1
        
        Parameters
        ----------
        superchannel : qutip.Qobj
            The superchannel operator
        input_channel : qutip.Qobj
            The input channel operator
            
        Returns
        -------
        qutip.Qobj
            The output channel
        """
        return (tensor(qeye(2), input_channel, qeye(2)).trans().trans() * 
                superchannel).ptrace([3, 0])
    
    @staticmethod
    def choi_state_unitary(v):
        """
        Create the Choi state of a unitary operator.
        
        The Choi state is defined as:
        J(Φ) = (Φ ⊗ I)(|Φ+⟩⟨Φ+|)
        where |Φ+⟩ = (|00⟩ + |11⟩)/√2 is the maximally entangled state.
        
        Parameters
        ----------
        v : qutip.Qobj
            Unitary operator
            
        Returns
        -------
        qutip.Qobj
            Choi state (4x4 matrix with dims [[2,2],[2,2]])
        """
        # Create the maximally entangled state |Φ+⟩
        ket_00 = tensor(basis(2, 0), basis(2, 0))
        ket_11 = tensor(basis(2, 1), basis(2, 1))
        phi_plus = (ket_00 + ket_11)
        
        choi = tensor(v, qeye(2)) * phi_plus
        choi = choi * choi.dag()
        choi.dims = [[2, 2], [2, 2]]
        
        return choi
    
    @staticmethod
    def parameterised_unitary(theta, phi, psi):
        """
        Create a parameterized unitary operator.
        
        V = rz(psi) * ry(phi) * rz(theta)
        
        Parameters
        ----------
        theta : float
            First rotation angle around z-axis
        phi : float
            Rotation angle around y-axis
        psi : float
            Second rotation angle around z-axis
            
        Returns
        -------
        qutip.Qobj
            Parameterized unitary operator
        """
        return QuantumChannelAnalyzer.rz(psi) * \
               QuantumChannelAnalyzer.ry(phi) * \
               QuantumChannelAnalyzer.rz(theta)
    
    @staticmethod
    def rz(theta):
        """
        Create a z-axis rotation operator.
        
        Parameters
        ----------
        theta : float
            Rotation angle
            
        Returns
        -------
        qutip.Qobj
            Rotation operator
        """
        return (-1j * theta / 2 * sigmaz()).expm()
    
    @staticmethod
    def ry(phi):
        """
        Create a y-axis rotation operator.
        
        Parameters
        ----------
        phi : float
            Rotation angle
            
        Returns
        -------
        qutip.Qobj
            Rotation operator
        """
        return (-1j * phi / 2 * sigmay()).expm()
    
    @staticmethod
    def _project_to_unitary(A):
        """
        Project a matrix to the nearest unitary using SVD.
        
        Parameters
        ----------
        A : qutip.Qobj
            Input operator
            
        Returns
        -------
        qutip.Qobj
            Nearest unitary operator
        """
        M = A.full()
        U, _, Vh = np.linalg.svd(M)
        Uu = U @ Vh
        return Qobj(Uu, dims=A.dims)
    
    @staticmethod
    def closest_unitary_channel(choi_output):
        """
        Find the closest unitary channel to a given channel (in Choi form).
        
        This maximizes the Frobenius norm fidelity between the given channel
        and a unitary channel.
        
        Parameters
        ----------
        choi_output : qutip.Qobj
            Choi state of the output channel
            
        Returns
        -------
        tuple
            (U, F_U) where U is the closest unitary and F_U is the fidelity
        """
        # Normalize Choi state
        J = choi_output / choi_output.tr()
        
        # Get dominant eigenvector (best pure approximation)
        eigvals, eigvecs = J.eigenstates()
        vec = eigvecs[int(np.argmax(eigvals))]
        
        # Reshape into an operator A
        d = int(np.sqrt(vec.shape[0]))
        A = Qobj(vec.full().reshape(d, d), dims=[[d], [d]])
        
        # Project A to the nearest unitary
        U = QuantumChannelAnalyzer._project_to_unitary(A)
        
        # Calculate fidelity to that unitary channel
        J_U = QuantumChannelAnalyzer.choi_state_unitary(U)
        F_U = fidelity(J, J_U / J_U.tr())
        
        return U, F_U