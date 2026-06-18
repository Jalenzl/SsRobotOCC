$stepFile = "E:\SsRobotOCC\backend\tests\fixtures\cad\plate_with_slot_100.step"
$url = "http://localhost:8000/api/v1/stp/hierarchy/convert?mode=hierarchy&linear_deflection=0.1&per_face=true"

$response = Invoke-WebRequest -Uri $url -Method POST -ContentType "application/octet-stream" -Body ([System.IO.File]::ReadAllBytes($stepFile)) -TimeoutSec 60
$response.StatusCode
$response.Content.Length
