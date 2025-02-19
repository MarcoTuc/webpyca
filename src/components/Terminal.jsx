function Terminal({ isLoading, error, result }) {
    return (
        <div style={{ marginTop: '10px' }}>
            {isLoading ? (
                <p>Loading Python environment...</p>
            ) : error ? (
                <p style={{ color: 'red' }}>Error: {error}</p>
            ) : (
                <pre style={{ 
                    backgroundColor: '#f5f5f5',
                    padding: '10px',
                    borderRadius: '4px'
                }}>
                    {result}
                </pre>
            )}
        </div>
    );
}

export default Terminal;