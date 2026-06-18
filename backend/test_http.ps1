try {
    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromSeconds(5)
    $response = $client.GetAsync("http://127.0.0.1:8000/health").GetAwaiter().GetResult()
    Write-Output ("Status: " + $response.StatusCode)
    $content = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    Write-Output ("Body: " + $content)
} catch {
    Write-Output ("ERR: " + $_.Exception.Message)
}
