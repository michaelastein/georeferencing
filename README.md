MATLAB and Python source code for Direct Georeferencing of images taken by Drones

Link to paper: https://www.mdpi.com/1424-8220/22/2/604

APA citation: 

Correia, C. A., Andrade, F. A., Sivertsen, A., Guedes, I. P., Pinto, M. F., Manhães, A. G., & Haddad, D. B. (2022). Comprehensive Direct Georeferencing of Aerial Images for Unmanned Aerial Systems Applications. Sensors, 22(2), 604.

Bibtex citation:

@article{correia2022comprehensive,
  title={Comprehensive Direct Georeferencing of Aerial Images for Unmanned Aerial Systems Applications},
  author={Correia, Carlos AM and Andrade, Fabio AA and Sivertsen, Agnar and Guedes, Ihannah Pinto and Pinto, Milena Faria and Manh{\~a}es, Aline Gesualdi and Haddad, Diego Barreto},
  journal={Sensors},
  volume={22},
  number={2},
  pages={604},
  year={2022},
  publisher={Multidisciplinary Digital Publishing Institute}
}



Annotate images with geo information:


# Make sure your terminal is in the folder with images and CSV
$csvFile = "2025_08_21-03_27_13_PM_filtered.csv"

# Import the CSV (comma-separated)
$rows = Import-Csv $csvFile

foreach ($row in $rows) {

    $imageFile = $row.wiris_image.Trim()
    
    if (-not $imageFile) {
        Write-Host "Skipping row: image filename missing"
        continue
    }
    
    if (-not (Test-Path $imageFile)) {
        Write-Host "File not found: $imageFile"
        continue
    }

    # Determine GPS refs
    $latRef = if ([double]$row.Latitude -ge 0) { "N" } else { "S" }
    $lonRef = if ([double]$row.Longitude -ge 0) { "E" } else { "W" }

    # Build ImageDescription string
    $description = "Yaw=$($row.GimbalYawE), Pitch=$($row.pitch_agisoft), Roll=$($row.roll), RelativeAlt=$($row.TAlt)"

    Write-Host "Processing $imageFile ..."

    # Annotate image with ExifTool (wrap numeric values in quotes)
    exiftool `
        "-GPSLatitude=$($row.Latitude)" `
        "-GPSLatitudeRef=$latRef" `
        "-GPSLongitude=$($row.Longitude)" `
        "-GPSLongitudeRef=$lonRef" `
        "-GPSAltitude=$($row.alt)" `
        "-GPSAltitudeRef=0" `
        "-ImageDescription=$description" `
        -overwrite_original "$imageFile"
}
