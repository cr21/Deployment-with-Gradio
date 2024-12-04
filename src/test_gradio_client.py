#test client
from gradio_client import Client, handle_file
import concurrent.futures
import time
from pathlib import Path

def make_prediction(client, image_url):
    """Make a single prediction"""
    try:
        result = client.predict(
            image_list=handle_file(image_url),
            # images=handle_file(image_url),
            api_name="/predict"
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Single test image URL
    #image_url = "https://media.istockphoto.com/id/1365977387/photo/ramen-with-steaming-sizzle.jpg?s=1024x1024&w=is&k=20&c=jK4kEg7caCNs45Eg47umG_oM6US7QqqLPR0udbh6F_Y="
    image_url = 'https://media.istockphoto.com/id/860541574/photo/cheesecake-with-fresh-blueberries.jpg?s=612x612&w=0&k=20&c=2IiBNCQv1x2Bwt9wTFuL87KTVpwV0qRavFvtcJIpsBw='
    # Initialize client
    client = Client("http://127.0.0.1:7860/")
    
    print("\\nSending 16 concurrent requests with the same image...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor to send 16 requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(make_prediction, client, image_url) 
            for _ in range(32)
        ]
        
        # Collect results as they complete
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                print(f"Completed prediction {i+1}/16")
            except Exception as e:
                print(f"Error in request {i+1}: {str(e)}")
    
    end_time = time.time()
    
    # Print results
    print(f"\nAll predictions completed in {end_time - start_time:.2f} seconds")
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\nRequest {i+1}:")
        print(result)

if __name__ == "__main__":
    main() 
 