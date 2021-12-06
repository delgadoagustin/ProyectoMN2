from mpi4py import MPI
from Bio import SeqIO
import numpy as np
from time import process_time ##MEDICION TIEMPO

inicio_total = process_time() ##MEDICION TIEMPO

## RUTAS DE ARCHIVOS
rutaGEN = "Archivos/genMN2.fasta"
rutaENTRADA = "Archivos/Virus_secuenciados.fasta"
rutaSALIDA = "Archivos/Salida.fasta"

## MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if rank == 0:
    ## LECTURA GEN
    inicio_lectura = process_time() ##MEDICION TIEMPO
    for secuencia in SeqIO.parse(rutaGEN,"fasta"):
        genM2 = secuencia

    ## LECTURA DATOS
    base = []
    for secuencia in SeqIO.parse(rutaENTRADA,"fasta"):
        base.append(secuencia)
    base = np.array(base,SeqIO.SeqRecord)
    base = np.array_split(base,size) ##Divide los datos en la cantidad de procesos
    fin_lectura = process_time() ##MEDICION TIEMPO
else:
    genM2 = None
    base = None

inicio_paralelo = process_time() ##MEDICION TIEMPO
## BROADCAST
genM2 = comm.bcast(genM2,root=0)
## SCATTER
base = comm.scatter(base,root=0)

## ANALISIS DE DATOS
resultados = []
for secuencia in base:
    if genM2.seq in secuencia.seq:
        resultados.append(secuencia)
fin_paralelo = process_time() ##MEDICION TIEMPO

resultados = comm.gather(resultados,root=0)
comm.reduce(inicio_paralelo,op=MPI.MAX,root=0)
comm.reduce(fin_paralelo,op=MPI.MAX,root=0)
## RESULTADOS
if rank == 0:
    inicio_muestreo = process_time() ##MEDICION TIEMPO
    ## La siguiente linea convierte la 'lista de listas' de resultados en una sola lista 
    resultados = [secuencia for l in resultados for secuencia in l]
    print("Coincidencias:",len(resultados),"\n\nIDENTIFICADORES: \n")
    for secuencia in resultados:  
        print(secuencia.description)
    
    ## GUARDADO DE RESULTADOS
    SeqIO.write(resultados,rutaSALIDA,"fasta")
    fin_total = process_time() ##MEDICION TIEMPO

    ##MEDICIONES DE TIEMPO
    print("\nTIEMPOS")
    print("Tiempo de LECTURA:",fin_lectura-inicio_lectura)
    print("Tiempo de PARALELO:",fin_paralelo-inicio_paralelo)
    print("Tiempo de MUESTREO:",fin_total-inicio_muestreo)
    print("Tiempo TOTAL:",fin_total-inicio_total)