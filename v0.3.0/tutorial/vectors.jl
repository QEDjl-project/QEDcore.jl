# # Vector and Matrix Types

# **TBW**

using QEDcore

# ## Lorentz Vectors

lv = rand(SLorentzVector)

# ## Bispinors and Adjoint Bispinors

bs = rand(BiSpinor)
#
abs = rand(AdjointBiSpinor)
#
abs * bs

# ## Dirac and Gamma Matrices

gm = rand(DiracMatrix)
#
abs * gm * bs
