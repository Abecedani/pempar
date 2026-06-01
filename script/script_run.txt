UNTUK MENJALANKAN PROGRAM SEKALIGUS, DAPAT MENGGUNAKAN SCRIPT DI BAWAH INI
for ($i = 1; $i -le 5; $i++) {
    Write-Host "=== RUN KE-$i ==="
    foreach ($dataset in @("1143", "50000", "100000", "500000", "1000000")) {
        Set-Content -Path "config.txt" -Value $dataset
        Write-Host "--- Dataset: $dataset ---"
        mpiexec -n 1 python script_mpi.py
        mpiexec -n 2 python script_mpi.py
        mpiexec -n 4 python script_mpi.py
    }
}