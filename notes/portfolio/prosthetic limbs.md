### Prosthetic Feet:

**Design motivation:**

the proper choice of foot is critical to reduce the negative effects of limb loss that include:

+ temporal and ground reaction force asymmetry

+ higher metabolic cost

+ increased prevalence of intact knee osteoarthritis



拓扑优化 (Topology Optimization)

1）**均匀化方法**

结构拓扑优化均匀化方法在采用有限元方法对设计区域进行离散化的基础上，将整个设计空间假设成类似“气孔分布”的微结构单元（单胞），单胞在优化开始前分布均匀，大小相同。在拓扑优化过程中，单胞密度分布发生变化，即高应力区域单胞密度变大，低应力区域单胞密度变小。优化过程中形成了一种承重结构，这种结构在高应力区域气孔”密集”，在低应力区域气孔密度较低。当迭代计算全部完成后，再定义一个合理的密度最小值，然后剔除设计空间中单胞密度低于这个最小值的区域，将产生一个材料效应最高的重量优化承重结构。

2）**变密度法**

变密度方法以连续变量的密度函数形式显式地表达单元相对密度与材料弹性模量之间的对应关系，寻求结构最佳 的传力路线，以实现优化设计区域内的材料分布，具有程序易实现、计算效率快、计算精度高的优势。