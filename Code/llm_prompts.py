
def write_prompt(problem, role, agent):
    match problem:
        case "synthetic_rubber" | "experimental_rubber" | "experimental_brain":
            match role:
                case "system":
                    match agent:
                        case "creator":
                            return _write_isotropic_system_creator_prompt()
                        case "inspector":
                            return _write_isotropic_system_inspector_prompt()
                        case _:
                            raise ValueError(f"Agent '{agent}' is not implemented for role '{role}' for problem '{problem}'.")
                case "user":
                    match agent:
                        case "creator":
                            return _write_isotropic_user_creator_prompt()
                        case "inspector":
                            return _write_isotropic_user_inspector_prompt()
                        case _:
                            raise ValueError(f"Agent '{agent}' is not implemented for role '{role}' for problem '{problem}'.")
                case _:
                    raise ValueError(f"Role '{role}' is not implemented for problem '{problem}'.")
        case _:
            raise ValueError(f"Problem '{problem}' is not implemented.")


def _write_isotropic_system_creator_prompt():
    return '''You are an expert in constitutive modeling and artificial neural networks. Your task is to assist in generating a Python script that implements a Constitutive Artificial Neural Network (CANN) for a hyperelastic incompressible isotropic material. The script will compute the stress from the deformation gradient using the principles of continuum mechanics and neural networks. Follow the provided skeleton and guidelines carefully.'''


def _write_isotropic_system_inspector_prompt():
    return '''You are an expert in constitutive modeling and artificial neural networks. Your task is to verify the adherence to physical constraints of a generated Python script that implements a Constitutive Artificial Neural Network (CANN) for a hyperelastic incompressible isotropic material. The script will compute the stress from the deformation gradient using the principles of continuum mechanics and neural networks. Follow the provided guidelines carefully.'''


def _write_isotropic_user_creator_prompt():
    return '''
1.	Your task is to complete a Python script that implements a constitutive model predicting the strain energy density psi and the first Piola–Kirchhoff stress P from the deformation gradient F for a hyperelastic, incompressible, isotropic material using a Constitutive Artificial Neural Network (CANN).
2.	Your model must adhere to the following physical constraints:
2a.	Thermodynamic consistency: Use one or more neural-network blocks to predict the scalar strain-energy density psi. Obtain the isochoric part of the stress P by differentiating psi with respect to the deformation. Complement the isochoric part with the volumetric part of P based on the hydrostatic pressure p, which serves as a Lagrange multiplier enforcing incompressibility. Determine p from the boundary condition that the normal stress in the third spatial direction vanishes.
2b.	Symmetry of the stress tensor: Use the invariants of C as network inputs instead of the components of F or C directly.
2c.	Objectivity: Guaranteed by using the invariants of C as network inputs (see above).
2d.	Material symmetry: Guaranteed by using the invariants of C as network inputs (see above).
2e.	Polyconvexity: Preserve polyconvexity through the network. Constrain all weights to be non-negative. All activation functions must be convex and monotonically increasing. Additionally, they must be at least twice continuously differentiable. Suitable choices include (parametric) Softplus, Exponential, Squared Softplus, Gaussian-CDF integral, and Smooth ReLU variants.
2f.	Growth condition: Guaranteed automatically under incompressibility.
2g.	Energy normalization: Subtract psi evaluated at C = I from the network output so that psi(C = I) = 0.
2h.	Stress normalization: Guaranteed automatically via the Lagrange multiplier p under isotropic incompressibility.
2i.	Non-negativity of strain energy: Guaranteed automatically by the preceding constraints.
3.	Consider the following implementation hints:
3a.	The deformation gradient F serving as input will always be provided as a tf.Tensor of dtype tf.float32 and shape (batch_size,3,3). The model output must be a dictionary {"P": P, "Psi": psi}. The output psi is a tf.Tensor of dtype tf.float32 and shape (batch_size,1), while the output P is a tf.Tensor of dtype tf.float32 and shape (batch_size,3,3).
3b.	Automatic differentiation of psi can be cleanly accomplished in TensorFlow using tf.GradientTape. Open a tf.GradientTape() context, use tape.watch() to track the necessary tensors, and ensure all operations within the block form a differentiable chain from input to output. Then obtain the gradient using tape.gradient(). Note that tf.GradientTape only works with eager execution and cannot differentiate through Keras symbolic placeholders.
3c.	You are not allowed to use static methods in your implementation.
4.	Take your time to plan your implementation step by step.
5.	Please respond with a completion of the Python script skeleton provided below. Include the marker <BEGIN PYTHON SCRIPT> at the beginning of the code and <END PYTHON SCRIPT> at the end. Your response will be automatically parsed to extract the Python script between these markers. Please strictly follow the structure and function signatures provided in the skeleton.


<BEGIN PYTHON SCRIPT>
import tensorflow as tf


class CANN(tf.keras.Model):
    def __init__(self, **kwargs):
        """ 
        Args:
            kwargs: Additional keyword arguments for the model initialization.
        """

    def psi_from_F(self, F):
        """
        Pure forward computation of psi from F. No tf.GradientTape inside.

        Args:
            F (tf.Tensor): Input tensor containing the deformation gradient F. Shape: (batch_size,3,3).
        Returns:
            tf.Tensor: Strain energy density psi predicted from the deformation gradient F. Shape: (batch_size,1)
        """

    def call(self, F):
        """
        Uses method psi_from_F and tf.GradientTape to predict both psi and P from F.

        Args:
            F (tf.Tensor): Input tensor containing the deformation gradient F. Shape: (batch_size,3,3)
        Returns:
            dict: Dictionary with keys "P" and "Psi":
                  - "P": First Piola-Kirchhoff stress tensor. Shape: (batch_size,3,3).
                  - "Psi": Strain energy density. Shape: (batch_size,1).
        """


def build_cann_model():
    """ Builds a CANN model for predicting the strain energy density psi and the first Piola Kirchhoff stress P from the deformation gradient F for hyperelastic, incompressible, isotropic materials.
    Args:
        -- None --
    Returns:
        CANN: A CANN model instance (subclass of tf.keras.Model).
    """


<END PYTHON SCRIPT>
    
6. '''


def _write_isotropic_user_inspector_prompt():
    return '''
1.	Your task is to inspect the Python script below that implements a constitutive model predicting the strain energy density psi and the first Piola–Kirchhoff stress P from the deformation gradient F for a hyperelastic, incompressible, isotropic material using a Constitutive Artificial Neural Network (CANN).
2.	Your task is to verify the adherence of the model defined in the script to the following physical constraints:
2a.	Thermodynamic consistency: The stress must be computed as the derivative of a scalar energy potential, not predicted independently. Verify that the network has a single scalar output psi, and that the stress is obtained via automatic differentiation of that output with respect to the deformation. The incompressibility constraint is enforced through a Lagrange multiplier pressure p to be determined from the boundary condition that the normal stress in the third spatial direction vanishes.
2b.	Symmetry of the stress tensor: The stress tensor must be symmetric to satisfy conservation of angular momentum. This is guaranteed if the network input is formulated only in terms of invariants of C. Under incompressibility, I3 = 1 is constant. Verify that the network takes exactly two inputs: I1 = tr(C) and I2 = tr(cof(C)), and that no other quantities enter the network. Note that under the incompressibility constraint det(F) = 1, it holds that I1​ >= 3 and I2​ >= 3 for all admissible deformations.
2c.	Objectivity: The energy must be unchanged when a rigid rotation is superimposed on the current configuration. This is ensured by the same criterion as point 2.
2d.	Material symmetry: The energy must be unchanged under any rotation of the reference configuration. This is ensured by the same criterion as point 2.
2e.	Polyconvexity: Polyconvexity implies material stability and prevents unphysical behavior in boundary value problems. Verify three aspects: (a) all weights in the network are constrained to be non-negative, (b) all activation functions are convex and monotonically increasing over the admissible input domain, and (c) all activation functions are at least twice continuously differentiable. If any weight is negative or any activation function violates convexity, monotonicity, or smoothness, polyconvexity is not guaranteed.
2f.	Growth condition: The energy must tend to infinity as volume tends to zero or infinity. The incompressibility constraint automatically guarantees this. 
2g.	Energy normalization: The energy must be zero in the undeformed configuration. Verify that the network output includes a subtraction of psi evaluated at C = 1, so that the final energy is zero in the undeformed state.
2h.	Stress normalization: The material must be stress-free in the undeformed configuration. The combination of isotropy and the incompressibility constraint automatically guarantees this.
2i.	Non-negativity of strain energy: The energy must be non-negative for all admissible deformation states. The combination of polyconvexity, energy normalization, and stress normalization automatically guarantees this.
3.  Your task is to inspect the provided constitutive model for compliance with the specified constraints yourself whenever possible. Use a corresponding numerical validation tool only if you are genuinely uncertain whether a specific constraint is fulfilled.
4.	Below is the CANN you are to inspect. Provide your feedback in the following JSON form: {"Thermodynamic consistency fulfilled": false, "Thermodynamic consistency explanation": "", "Symmetry of the stress tensor fulfilled": false, "Symmetry of the stress tensor explanation": "", "Objectivity fulfilled": false, "Objectivity explanation": "", "Material symmetry fulfilled": false, "Material symmetry explanation": "", "Polyconvexity fulfilled": false, "Polyconvexity explanation": "", "Growth condition fulfilled": false, "Growth condition explanation": "", "Energy normalization fulfilled": false, "Energy normalization explanation": "", "Stress normalization fulfilled": false, "Stress normalization explanation": "", "Non-negativity of strain energy fulfilled": false, "Non-negativity of strain energy explanation": ""}. For each constraint fulfillment, provide exactly one of the two possible values: True if the condition is fulfilled, and False if the condition is violated. In addition, provide your explanation on why this condition is fulfilled or violated for each constraint as a brief text. Please strictly follow this form.

'''
