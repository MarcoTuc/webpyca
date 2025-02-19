import { useRef, useEffect } from "react"
import p5 from "p5"


function P5Wrapper({ sketch, id }) {
    
    const p5ContainerRef = useRef();

    useEffect(() => {
        const p5Instance = new p5(sketch, p5ContainerRef.current);

        return () => {
            p5Instance.remove();
        }
    }, [sketch]); // Add sketch to dependency array

    return (
        <div 
            className="P5Canvas" 
            id = {id}
            ref = {p5ContainerRef}
        />
    );
}

export default P5Wrapper