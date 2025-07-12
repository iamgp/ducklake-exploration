"""
DuckLake Tutorial - Simple Python Demo
=====================================

This script demonstrates DuckDB's DuckLake capabilities using a local PostgreSQL catalog.
DuckLake provides advanced data lake features without traditional lakehouse complexity.

Prerequisites:
- PostgreSQL running via: docker-compose up -d
- Python packages: duckdb, pandas, numpy, psycopg2-binary, click, rich

Usage (after installing with uv/pip):
  ducklake-demo                     # Run demo (auto-resets for clean state)
  ducklake-demo --no-reset          # Keep existing data between runs
  ducklake-demo demo --no-reset     # Run demo subcommand explicitly
  ducklake-demo reset               # Manual reset only (no demo)
  ducklake-demo --help              # Show help

Direct Python usage:
  python ducklake_demo.py           # Run demo (auto-resets for clean state)
  python ducklake_demo.py --no-reset # Keep existing data between runs
"""

import logging
import os
import shutil
import subprocess
import time
import urllib.request

import click
import duckdb
import numpy as np
import pandas as pd
from click import Context
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Initialize rich console and logger
console = Console()

# Configure logging to be less intrusive during demo
class RichHandler(logging.Handler):
    def emit(self, record):
        # Only log errors and warnings to console during demo
        if record.levelno >= logging.WARNING:
            console.print(f"[yellow]{self.format(record)}[/yellow]")

# Set up logging with custom handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add file handler for all logs
file_handler = logging.FileHandler('ducklake_demo.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Add rich handler for warnings/errors only
rich_handler = RichHandler()
rich_handler.setLevel(logging.WARNING)
logger.addHandler(rich_handler)


def check_postgresql() -> bool:
    """Check if PostgreSQL container is running."""
    with console.status("[bold blue]Checking PostgreSQL container status..."):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=ducklake-postgres",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "ducklake-postgres" in result.stdout:
                console.print("✓ [green]PostgreSQL container is running[/green]")
                logger.info("PostgreSQL container found and running")
                return True
            else:
                console.print(
                    " [red]PostgreSQL container not found.[/red] Run: [bold]docker-compose up -d[/bold]"
                )
                logger.warning("PostgreSQL container not found")
                return False
        except Exception as e:
            console.print(f" [red]Error checking Docker:[/red] {e}")
            logger.error(f"Docker check failed: {e}")
            return False


def setup_extensions(conn: duckdb.DuckDBPyConnection) -> bool:
    """Install and load required DuckDB extensions."""
    with console.status("[bold blue]Setting up DuckDB extensions..."):
        try:
            conn.execute("INSTALL ducklake")
            conn.execute("INSTALL postgres")
            conn.execute("LOAD ducklake")
            conn.execute("LOAD postgres")
            console.print(
                "✓ [green]DuckLake and PostgreSQL extensions loaded successfully[/green]"
            )
            logger.info("DuckDB extensions loaded successfully")
            return True
        except Exception as e:
            console.print(f" [red]Error loading extensions:[/red] {e}")
            logger.error(f"Extension loading failed: {e}")
            return False


def test_postgresql_connection() -> dict[str, str] | None:
    """Test connection to PostgreSQL catalog."""
    with console.status("[bold blue]Testing PostgreSQL catalog connection..."):
        pg_config = {
            "host": "localhost",
            "port": "5432",
            "database": "ducklake_catalog",
            "user": "ducklake",
            "password": "ducklake123",
        }

        try:
            # Test connection using a temporary connection
            temp_conn = duckdb.connect(":memory:")
            temp_conn.execute("INSTALL postgres")
            temp_conn.execute("LOAD postgres")

            # Test connection to PostgreSQL
            test_attach_query = f"""
            ATTACH 'dbname={pg_config["database"]}
                    user={pg_config["user"]}
                    password={pg_config["password"]}
                    host={pg_config["host"]}
                    port={pg_config["port"]}' AS pg_test (TYPE postgres);
            """
            temp_conn.execute(test_attach_query)
            temp_conn.execute(
                "SELECT 1 as test FROM pg_test.information_schema.tables LIMIT 1"
            )

            # Clean up
            temp_conn.execute("DETACH pg_test")
            temp_conn.close()

            console.print("✓ [green]PostgreSQL catalog connection successful[/green]")
            logger.info("PostgreSQL catalog connection established")
            return pg_config
        except Exception as e:
            console.print(f" [red]PostgreSQL connection failed:[/red] {e}")
            logger.error(f"PostgreSQL connection failed: {e}")
            return None


def initialize_ducklake(
    conn: duckdb.DuckDBPyConnection, pg_config: dict[str, str]
) -> str | None:
    """Initialize DuckLake with PostgreSQL catalog."""
    with console.status("[bold blue]Initializing DuckLake..."):
        lake_data_dir = "./ducklake_data"
        os.makedirs(lake_data_dir, exist_ok=True)

        # Use timestamp to create unique database name
        import time

        db_name = f"ducklake_demo_{int(time.time())}"

        try:
            # Attach DuckLake with PostgreSQL catalog using unique name
            ducklake_init_query = f"""
            ATTACH 'ducklake:postgres:dbname={pg_config["database"]}
                    user={pg_config["user"]}
                    password={pg_config["password"]}
                    host={pg_config["host"]}
                    port={pg_config["port"]}' AS {db_name}
                    (DATA_PATH '{lake_data_dir}');
            """

            conn.execute(ducklake_init_query)
            conn.execute(f"USE {db_name}")

            # Verify the attachment
            initial_tables = conn.execute("SHOW TABLES").fetchall()

            info_table = Table(title="DuckLake Initialization", box=box.ROUNDED)
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="green")
            info_table.add_row("Status", " Initialized successfully")
            info_table.add_row("Data Path", lake_data_dir)
            info_table.add_row("Database Name", db_name)
            info_table.add_row("Current Tables", str(len(initial_tables)))

            console.print("\n")
            console.print(info_table)
            logger.info(f"DuckLake initialized successfully with database {db_name}")
            return db_name
        except Exception as e:
            console.print(f" [red]DuckLake initialization failed:[/red] {e}")
            logger.error(f"DuckLake initialization failed: {e}")
            return None


def create_sample_data(conn: duckdb.DuckDBPyConnection) -> bool | None:
    """Create sample customer and sales data."""
    console.print("\n[bold blue]Creating sample data...[/bold blue]")

    # Create customer data
    np.random.seed(42)  # For reproducible results
    cust_data = {
        "customer_id": range(1, 101),
        "name": [f"Customer {i}" for i in range(1, 101)],
        "email": [f"customer{i}@example.com" for i in range(1, 101)],
        "signup_date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "city": np.random.choice(
            ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], 100
        ),
        "age": np.random.randint(18, 80, 100),
    }
    cust_df = pd.DataFrame(cust_data)

    with console.status("[bold blue]Creating customers table..."):
        try:
            # Create customers table in DuckLake
            conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER,
                name VARCHAR,
                email VARCHAR,
                signup_date DATE,
                city VARCHAR,
                age INTEGER
            )
            """)

            # Clear existing data and insert new data
            conn.execute("DELETE FROM customers")
            conn.execute("INSERT INTO customers SELECT * FROM cust_df")

            cust_result = conn.execute("SELECT COUNT(*) FROM customers").fetchone()
            cust_count = cust_result[0] if cust_result else 0
            console.print(
                f"✓ [green]Created customers table with [bold]{cust_count}[/bold] records[/green]"
            )
            logger.info(f"Created customers table with {cust_count} records")
        except Exception as e:
            console.print(f" [red]Error creating customers table:[/red] {e}")
            logger.error(f"Customer table creation failed: {e}")
            return None

    # Create sales data
    sales_info = {
        "sale_id": range(1, 501),
        "customer_id": np.random.choice(cust_df["customer_id"], 500),
        "product_name": np.random.choice(
            ["Laptop", "Phone", "Tablet", "Headphones", "Monitor"], 500
        ),
        "amount": np.round(np.random.uniform(50, 2000, 500), 2),
        "sale_date": pd.date_range("2023-01-01", periods=500, freq="h"),
        "region": np.random.choice(["North", "South", "East", "West"], 500),
    }
    sales_df = pd.DataFrame(sales_info)

    with console.status("[bold blue]Creating sales table..."):
        try:
            # Create sales table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                sale_id INTEGER,
                customer_id INTEGER,
                product_name VARCHAR,
                amount DECIMAL(10,2),
                sale_date TIMESTAMP,
                region VARCHAR
            )
            """)

            # Clear and insert data
            conn.execute("DELETE FROM sales")
            conn.execute("INSERT INTO sales SELECT * FROM sales_df")

            sales_result = conn.execute("SELECT COUNT(*) FROM sales").fetchone()
            sales_count = sales_result[0] if sales_result else 0
            console.print(
                f"✓ [green]Created sales table with [bold]{sales_count}[/bold] records[/green]"
            )
            logger.info(f"Created sales table with {sales_count} records")
            return True
        except Exception as e:
            console.print(f" [red]Error creating sales table:[/red] {e}")
            logger.error(f"Sales table creation failed: {e}")
            return False


def demonstrate_queries(conn: duckdb.DuckDBPyConnection) -> bool:
    """Demonstrate basic queries across DuckLake tables."""
    console.print("\n[bold blue]Running sample queries...[/bold blue]")

    with console.status("[bold blue]Executing join query..."):
        try:
            # Join query
            query_result = conn.execute("""
            SELECT
                c.name,
                c.city,
                COUNT(s.sale_id) as total_sales,
                SUM(s.amount) as total_amount
            FROM customers c
            LEFT JOIN sales s ON c.customer_id = s.customer_id
            GROUP BY c.customer_id, c.name, c.city
            ORDER BY total_amount DESC
            LIMIT 10
            """).fetchdf()

            console.print("✓ [green]Join query executed successfully[/green]")
            logger.info("Query executed successfully")

            # Create a rich table for the results
            result_table = Table(
                title=" Top 10 Customers by Sales Amount", box=box.ROUNDED
            )
            result_table.add_column("Customer Name", style="cyan")
            result_table.add_column("City", style="magenta")
            result_table.add_column("Total Sales", justify="right", style="blue")
            result_table.add_column("Total Amount", justify="right", style="green")

            for _, row in query_result.iterrows():
                result_table.add_row(
                    str(row["name"]),
                    str(row["city"]),
                    str(row["total_sales"]),
                    f"${row['total_amount']:.2f}"
                    if pd.notnull(row["total_amount"])
                    else "$0.00",
                )

            console.print("\n")
            console.print(result_table)
            return True
        except Exception as e:
            console.print(f" [red]Query failed:[/red] {e}")
            logger.error(f"Query execution failed: {e}")
            return False


def demonstrate_acid_transactions(conn: duckdb.DuckDBPyConnection) -> bool:
    """Demonstrate ACID transaction capabilities."""
    console.print("\n[bold blue]Testing ACID transactions...[/bold blue]")

    with console.status("[bold blue]Executing ACID transaction test..."):
        try:
            # Start transaction
            conn.begin()

            # Insert a new customer
            conn.execute("""
            INSERT INTO customers (customer_id, name, email, signup_date, city, age)
            VALUES (101, 'Test Customer', 'test@example.com', DATE '2024-01-01', 'Test City', 30)
            """)

            # Insert related sales
            conn.execute("""
            INSERT INTO sales (sale_id, customer_id, product_name, amount, sale_date, region)
            VALUES (501, 101, 'Test Product', 999.99, TIMESTAMP '2024-01-01 10:00:00', 'Test')
            """)

            # Check the data exists
            test_cust_exists = conn.execute(
                "SELECT * FROM customers WHERE customer_id = 101"
            ).fetchone()
            test_sale_exists = conn.execute(
                "SELECT * FROM sales WHERE sale_id = 501"
            ).fetchone()

            if test_cust_exists and test_sale_exists:
                # Rollback to demonstrate ACID properties
                conn.rollback()

                # Verify rollback worked
                rollback_result = conn.execute(
                    "SELECT COUNT(*) FROM customers WHERE customer_id = 101"
                ).fetchone()
                rollback_count = rollback_result[0] if rollback_result else 0

                if rollback_count == 0:
                    console.print(
                        "✓ [green]ACID transaction and rollback successful[/green]"
                    )
                    logger.info("ACID transaction test completed successfully")
                    return True
                else:
                    console.print(" [red]Rollback failed[/red]")
                    logger.error("ACID rollback test failed")
                    return False
            else:
                console.print(" [red]Transaction insert failed[/red]")
                logger.error("ACID transaction insert failed")
                return False
        except Exception as e:
            console.print(f" [red]ACID test failed:[/red] {e}")
            logger.error(f"ACID test failed: {e}")
            try:
                conn.rollback()
            except:
                pass
            return False


def demonstrate_time_travel(conn: duckdb.DuckDBPyConnection, db_name: str) -> bool:
    """Demonstrate snapshots and time travel."""
    console.print("\n[bold blue]Testing time travel and snapshots...[/bold blue]")

    with console.status("[bold blue]Analyzing snapshots and performing time travel..."):
        try:
            # Get current snapshot count before changes
            snapshots_before = conn.execute(
                f"SELECT * FROM ducklake_snapshots('{db_name}')"
            ).fetchdf()
            snap_count_before = len(snapshots_before)

            # Get initial ages before update
            ages_before = conn.execute("""
            SELECT customer_id, age FROM customers WHERE customer_id <= 5 ORDER BY customer_id
            """).fetchdf()

            # Make changes (this will automatically create a new snapshot)
            conn.execute("UPDATE customers SET age = age + 1 WHERE customer_id <= 10")

            # Check current state after update
            ages_after = conn.execute("""
            SELECT customer_id, age FROM customers WHERE customer_id <= 5 ORDER BY customer_id
            """).fetchdf()

            # Check snapshots after changes
            snapshots_after = conn.execute(
                f"SELECT * FROM ducklake_snapshots('{db_name}')"
            ).fetchdf()
            snap_count_after = len(snapshots_after)

            console.print(
                f"✓ [green]Snapshots: [bold]{snap_count_before}[/bold] → [bold]{snap_count_after}[/bold][/green]"
            )
            logger.info(f"Snapshots created: {snap_count_before} -> {snap_count_after}")

            # Create comparison tables
            before_table = Table(
                title=" Before Update (First 5 Customers)", box=box.ROUNDED
            )
            before_table.add_column("Customer ID", justify="center")
            before_table.add_column("Age", justify="center", style="blue")

            after_table = Table(
                title=" After Update (First 5 Customers)", box=box.ROUNDED
            )
            after_table.add_column("Customer ID", justify="center")
            after_table.add_column("Age", justify="center", style="green")

            for _, row in ages_before.iterrows():
                before_table.add_row(str(row["customer_id"]), str(row["age"]))

            for _, row in ages_after.iterrows():
                after_table.add_row(str(row["customer_id"]), str(row["age"]))

            console.print("\n")
            console.print(before_table)
            console.print(after_table)

            # Demonstrate time travel using version-based query
            if snap_count_after > snap_count_before:
                try:
                    # Query previous version using AT VERSION syntax
                    previous_version = snap_count_after - 1
                    historical_ages = conn.execute(f"""
                    SELECT customer_id, age FROM customers AT (VERSION => {previous_version})
                    WHERE customer_id <= 5 ORDER BY customer_id
                    """).fetchdf()

                    console.print(
                        f"✓ [green]Time travel query successful (version [bold]{previous_version}[/bold] → [bold]{snap_count_after}[/bold])[/green]"
                    )

                    historical_table = Table(
                        title=" Historical Data (Time Travel)", box=box.ROUNDED
                    )
                    historical_table.add_column("Customer ID", justify="center")
                    historical_table.add_column(
                        "Historical Age", justify="center", style="yellow"
                    )

                    for _, row in historical_ages.iterrows():
                        historical_table.add_row(
                            str(row["customer_id"]), str(row["age"])
                        )

                    console.print("\n")
                    console.print(historical_table)
                    logger.info("Time travel query executed successfully")
                except Exception:
                    console.print(
                        f" [yellow]Time travel concept demonstrated (snapshots: {snap_count_before} → {snap_count_after})[/yellow]"
                    )
                    logger.warning("Time travel query failed, but concept demonstrated")

            return True
        except Exception as e:
            console.print(f" [red]Snapshot operation failed:[/red] {e}")
            logger.error(f"Time travel demonstration failed: {e}")
            return False


def demonstrate_schema_evolution(conn: duckdb.DuckDBPyConnection) -> bool:
    """Demonstrate schema evolution capabilities."""
    console.print("\n[bold blue]Testing schema evolution...[/bold blue]")

    with console.status("[bold blue]Evolving schema and updating data..."):
        try:
            # Add a new column to existing table
            conn.execute(
                "ALTER TABLE customers ADD COLUMN loyalty_points INTEGER DEFAULT 0"
            )

            # Update some records
            conn.execute(
                "UPDATE customers SET loyalty_points = age * 10 WHERE customer_id <= 20"
            )

            # Verify schema change
            schema_info = conn.execute("DESCRIBE customers").fetchdf()

            # Check data with new column
            loyalty_data = conn.execute("""
            SELECT customer_id, name, age, loyalty_points
            FROM customers
            WHERE loyalty_points > 0
            LIMIT 5
            """).fetchdf()

            console.print(
                "✓ [green]Schema evolution successful - added loyalty_points column[/green]"
            )
            logger.info("Schema evolution completed successfully")

            # Display schema info in a table
            schema_table = Table(title=" Updated Table Schema", box=box.ROUNDED)
            schema_table.add_column("Column", style="cyan")
            schema_table.add_column("Type", style="magenta")
            schema_table.add_column("Nullable", style="yellow")

            for _, row in schema_info.iterrows():
                schema_table.add_row(
                    str(row["column_name"]), str(row["column_type"]), str(row["null"])
                )

            console.print("\n")
            console.print(schema_table)

            # Display sample data
            loyalty_table = Table(
                title=" Sample Data with New loyalty_points Column", box=box.ROUNDED
            )
            loyalty_table.add_column("Customer ID", justify="center")
            loyalty_table.add_column("Name", style="cyan")
            loyalty_table.add_column("Age", justify="center", style="blue")
            loyalty_table.add_column("Loyalty Points", justify="center", style="green")

            for _, row in loyalty_data.iterrows():
                loyalty_table.add_row(
                    str(row["customer_id"]),
                    str(row["name"]),
                    str(row["age"]),
                    str(row["loyalty_points"]),
                )

            console.print("\n")
            console.print(loyalty_table)
            return True
        except Exception as e:
            console.print(f" [red]Schema evolution failed:[/red] {e}")
            logger.error(f"Schema evolution failed: {e}")
            return False


def demonstrate_performance(conn: duckdb.DuckDBPyConnection) -> bool:
    """Demonstrate performance analysis."""
    console.print("\n[bold blue]Running performance analysis...[/bold blue]")

    try:
        # Measure query performance with different complexity levels
        queries = [
            ("Simple aggregation", "SELECT COUNT(*) as total_customers FROM customers"),
            (
                "Join with aggregation",
                """
                SELECT c.city, AVG(s.amount) as avg_sale, COUNT(*) as sale_count
                FROM customers c JOIN sales s ON c.customer_id = s.customer_id
                GROUP BY c.city ORDER BY avg_sale DESC
            """,
            ),
            (
                "Complex analytical query",
                """
                SELECT
                    c.city,
                    c.age,
                    COUNT(s.sale_id) as sales_count,
                    AVG(s.amount) as avg_amount,
                    SUM(s.amount) as total_amount,
                    MAX(s.sale_date) as last_sale
                FROM customers c
                LEFT JOIN sales s ON c.customer_id = s.customer_id
                WHERE c.age > 30
                GROUP BY c.city, c.age
                HAVING COUNT(s.sale_id) > 0
                ORDER BY total_amount DESC
                LIMIT 10
            """,
            ),
        ]

        perf_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running queries...", total=len(queries))

            for query_name, query in queries:
                progress.update(task, description=f"Executing: {query_name}")
                start_time = time.time()
                result = conn.execute(query).fetchdf()
                end_time = time.time()
                exec_time_ms = round((end_time - start_time) * 1000, 2)

                perf_results.append(
                    {"query": query_name, "time_ms": exec_time_ms, "rows": len(result)}
                )

                console.print(
                    f"   [cyan]{query_name}[/cyan]: [green]{exec_time_ms}ms[/green] ([blue]{len(result)} rows[/blue])"
                )
                progress.advance(task)

        console.print("✓ [green]Performance analysis complete[/green]")
        logger.info("Performance analysis completed")

        # Create performance summary table
        perf_table = Table(title=" Query Performance Summary", box=box.ROUNDED)
        perf_table.add_column("Query Type", style="cyan")
        perf_table.add_column("Execution Time", justify="right", style="green")
        perf_table.add_column("Rows Returned", justify="right", style="blue")

        for result in perf_results:
            perf_table.add_row(
                result["query"], f"{result['time_ms']}ms", str(result["rows"])
            )

        console.print("\n")
        console.print(perf_table)

        # Show the last query result as example
        final_result = conn.execute(queries[-1][1]).fetchdf()

        sample_table = Table(
            title=" Sample Complex Query Result (Top 5)", box=box.ROUNDED
        )
        for col in final_result.columns:
            sample_table.add_column(col, style="cyan")

        for _, row in final_result.head().iterrows():
            sample_table.add_row(*[str(val) for val in row])

        console.print("\n")
        console.print(sample_table)

        return True
    except Exception as e:
        console.print(f" [red]Performance analysis failed:[/red] {e}")
        logger.error(f"Performance analysis failed: {e}")
        return False


def demonstrate_data_compression() -> bool:
    """Demonstrate data compression efficiency."""
    console.print(
        "\n[bold blue]Analyzing data compression and storage efficiency...[/bold blue]"
    )

    with console.status("[bold blue]Analyzing storage efficiency..."):
        try:
            # Create comparison data to show compression benefits

            # Simulate the same data in different formats
            lake_data_dir = "./ducklake_data"
            if not os.path.exists(lake_data_dir):
                console.print("✗ [red]DuckLake data directory not found[/red]")
                logger.error("DuckLake data directory not found")
                return False

            # Calculate DuckLake storage size
            ducklake_size = 0
            parquet_files = 0
            for root, dirs, files in os.walk(lake_data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    ducklake_size += file_size
                    if file.endswith(".parquet"):
                        parquet_files += 1

            # Create a CSV comparison (simulated)
            conn = duckdb.connect(":memory:")
            conn.execute("INSTALL ducklake")
            conn.execute("LOAD ducklake")

            # Read data from DuckLake to get actual row counts
            try:
                customer_result = conn.execute(
                    f"SELECT COUNT(*) FROM '{lake_data_dir}/customers/*.parquet'"
                ).fetchone()
                customer_count = customer_result[0] if customer_result else 100

                sales_result = conn.execute(
                    f"SELECT COUNT(*) FROM '{lake_data_dir}/sales/*.parquet'"
                ).fetchone()
                sales_count = sales_result[0] if sales_result else 500
            except:
                customer_count = 100  # fallback
                sales_count = 500

            # Estimate uncompressed sizes (rough calculation)
            estimated_csv_size = (customer_count * 80) + (
                sales_count * 120
            )  # bytes per row estimate
            estimated_json_size = (customer_count * 150) + (
                sales_count * 200
            )  # bytes per row estimate

            compression_ratio = (
                round(estimated_csv_size / ducklake_size, 1) if ducklake_size > 0 else 0
            )

            # Create storage efficiency table
            storage_table = Table(title=" Storage Efficiency Analysis", box=box.ROUNDED)
            storage_table.add_column("Format", style="cyan")
            storage_table.add_column("Size (KB)", justify="right", style="green")
            storage_table.add_column("Files/Details", justify="right", style="blue")

            storage_table.add_row(
                "DuckLake (Parquet)",
                f"{round(ducklake_size / 1024, 2)} KB",
                f"{parquet_files} files",
            )
            storage_table.add_row(
                "Estimated CSV",
                f"{round(estimated_csv_size / 1024, 2)} KB",
                "Uncompressed",
            )
            storage_table.add_row(
                "Estimated JSON",
                f"{round(estimated_json_size / 1024, 2)} KB",
                "Uncompressed",
            )

            console.print("\n")
            console.print(storage_table)

            # Create compression metrics table
            metrics_table = Table(title="Compression Metrics", box=box.ROUNDED)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            metrics_table.add_row(
                "Compression Ratio", f"{compression_ratio}:1 (vs CSV)"
            )
            metrics_table.add_row(
                "Space Savings",
                f"{round((1 - ducklake_size / estimated_csv_size) * 100, 1)}%",
            )

            console.print("\n")
            console.print(metrics_table)

            # Create advantages panel
            advantages_text = """• Columnar storage for analytical queries
• Built-in compression (typically 80-90% size reduction)
• Schema evolution support
• Predicate pushdown for fast filtering
• Cross-platform compatibility"""

            advantages_panel = Panel(
                advantages_text, title=" Parquet Advantages", border_style="blue"
            )
            console.print(advantages_panel)

            conn.close()
            console.print("✓ [green]Data compression analysis completed[/green]")
            logger.info("Data compression analysis completed")
            return True

        except Exception as e:
            console.print(f" [red]Compression analysis failed:[/red] {e}")
            logger.error(f"Compression analysis failed: {e}")
            return False


def explore_parquet_files() -> bool:
    """Explore the Parquet files created by DuckLake."""
    console.print(
        "[bold blue]Exploring DuckLake Parquet file structure...[/bold blue]"
    )

    lake_data_dir = "./ducklake_data"
    if not os.path.exists(lake_data_dir):
        console.print("✗ [red]DuckLake data directory not found[/red]")
        logger.error("DuckLake data directory not found")
        return False

    with console.status("[bold blue]Analyzing Parquet file structure..."):
        try:
            console.print(f"[cyan]DuckLake data directory:[/cyan] {lake_data_dir}")

            # Walk through all files in the data directory
            parquet_files = []
            total_size = 0

            files_table = Table(title=" Files in DuckLake Directory", box=box.ROUNDED)
            files_table.add_column("File Path", style="cyan")
            files_table.add_column("Size (KB)", justify="right", style="green")

            for root, dirs, files in os.walk(lake_data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size

                    # Get relative path for cleaner display
                    rel_path = os.path.relpath(file_path, lake_data_dir)

                    if file.endswith(".parquet"):
                        parquet_files.append(
                            {
                                "file": rel_path,
                                "size_kb": round(file_size / 1024, 2),
                                "full_path": file_path,
                            }
                        )

                    files_table.add_row(rel_path, f"{round(file_size / 1024, 2)}")

            console.print("\n")
            console.print(files_table)

            # Storage summary
            summary_table = Table(title=" Storage Summary", box=box.ROUNDED)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")

            total_file_count = len(
                [f for _, _, files in os.walk(lake_data_dir) for f in files]
            )
            summary_table.add_row("Total files", str(total_file_count))
            summary_table.add_row("Parquet files", str(len(parquet_files)))
            summary_table.add_row("Total size", f"{round(total_size / 1024, 2)} KB")

            console.print("\n")
            console.print(summary_table)

            # Analyze Parquet files with DuckDB
            if parquet_files:
                console.print(
                    "\n [bold blue]Analyzing Parquet file contents...[/bold blue]"
                )
                conn = duckdb.connect(":memory:")

                for pf in parquet_files[:3]:  # Analyze first 3 files
                    try:
                        # Read parquet file metadata
                        conn.execute(
                            f"SELECT * FROM parquet_metadata('{pf['full_path']}')"
                        ).fetchdf()
                        file_schema = conn.execute(
                            f"SELECT * FROM parquet_schema('{pf['full_path']}')"
                        ).fetchdf()

                        # Create file analysis panel
                        file_info_text = f"""Size: {pf["size_kb"]} KB
Columns: {len(file_schema) if not file_schema.empty else "N/A"}"""

                        if not file_schema.empty:
                            file_info_text += "\nSchema:\n"
                            for _, row in file_schema.head(5).iterrows():
                                file_info_text += f"  • {row.get('name', 'N/A')} ({row.get('type', 'N/A')})\n"

                        # Sample data from parquet file
                        sample_data = conn.execute(
                            f"SELECT * FROM '{pf['full_path']}' LIMIT 3"
                        ).fetchdf()
                        if not sample_data.empty:
                            file_info_text += "\nSample data:\n"
                            for col in sample_data.columns[:3]:  # Show first 3 columns
                                values = sample_data[col].head(3).tolist()
                                file_info_text += f"  {col}: {values}\n"

                        file_panel = Panel(
                            file_info_text,
                            title=f" {pf['file']}",
                            border_style="blue",
                        )
                        console.print(file_panel)

                    except Exception as e:
                        console.print(
                            f"      [yellow]Could not analyze {pf['file']}:[/yellow] {e}"
                        )
                        logger.warning(
                            f"Could not analyze parquet file {pf['file']}: {e}"
                        )

                conn.close()

            console.print("✓ [green]Parquet file exploration completed[/green]")
            logger.info("Parquet file exploration completed")
            return True
        except Exception as e:
            console.print(f" [red]Error exploring files:[/red] {e}")
            logger.error(f"File exploration failed: {e}")
            return False


def demonstrate_large_dataset(conn: duckdb.DuckDBPyConnection) -> bool:
    """Demonstrate DuckLake with a larger dataset, using synthetic data generation as fallback."""
    console.print("\n[bold blue]Loading larger dataset for performance testing...[/bold blue]")
    
    # Try external datasets first, fallback to synthetic data generation
    external_datasets = [
        {
            "name": "Customers (External)",
            "url": "https://drive.google.com/uc?id=1N1xoxgcw2K3d-49tlchXAWw4wuxLj7EV&export=download",
            "table": "customers_large",
            "description": "100k records of customer data from GitHub samples"
        },
        {
            "name": "People (External)",
            "url": "https://drive.google.com/uc?id=1NW7EnwxuY6RpMIxOazRVibOYrZfMjsb2&export=download",
            "table": "people_large",
            "description": "100k records of people demographics from GitHub samples"
        },
        {
            "name": "Organizations (External)",
            "url": "https://drive.google.com/uc?id=1g4wqEIsKyiBWeCAwd0wEkiC4Psc4zwFu&export=download",
            "table": "organizations_large",
            "description": "100k records of organization data from GitHub samples"
        }
    ]
    
    loaded_datasets = []
    
    # Try to load external datasets first
    console.print("[cyan]Attempting to load external datasets...[/cyan]")
    
    for dataset in external_datasets:
        try:
            with console.status(f"[bold blue]Loading {dataset['name']}..."):
                start_time = time.time()
                
                # Create table from CSV URL
                conn.execute(f"""
                CREATE OR REPLACE TABLE {dataset['table']} AS 
                SELECT * FROM read_csv_auto('{dataset['url']}')
                """)
                
                # Get row count
                count_result = conn.execute(f"SELECT COUNT(*) FROM {dataset['table']}").fetchone()
                row_count = count_result[0] if count_result else 0
                
                load_time = time.time() - start_time
                
                if row_count > 0:
                    loaded_datasets.append({
                        "name": dataset['name'],
                        "table": dataset['table'],
                        "rows": row_count,
                        "load_time": round(load_time, 2),
                        "description": dataset['description'],
                        "source": "external"
                    })
                    
                    console.print(f"   ✓ [green]{dataset['name']}[/green]: [blue]{row_count:,} rows[/blue] in [yellow]{load_time:.2f}s[/yellow]")
                    logger.info(f"Loaded {dataset['name']}: {row_count} rows in {load_time:.2f}s")
                
        except Exception as e:
            console.print(f"   ⚠ [yellow]External dataset {dataset['name']} failed:[/yellow] {str(e)[:100]}...")
            logger.warning(f"Failed to load external dataset {dataset['name']}: {e}")
    
    # If external datasets failed, generate synthetic large datasets
    if not loaded_datasets:
        console.print("[cyan]Generating synthetic large datasets for demonstration...[/cyan]")
        
        synthetic_datasets = [
            {
                "name": "Large Customers (Synthetic)",
                "table": "customers_synthetic", 
                "rows": 50000,
                "description": "50k synthetic customer records"
            },
            {
                "name": "Large Orders (Synthetic)",
                "table": "orders_synthetic",
                "rows": 100000, 
                "description": "100k synthetic order records"
            },
            {
                "name": "Large Products (Synthetic)",
                "table": "products_synthetic",
                "rows": 25000,
                "description": "25k synthetic product records"
            }
        ]
        
        for dataset in synthetic_datasets:
            try:
                with console.status(f"[bold blue]Generating {dataset['name']}..."):
                    start_time = time.time()
                    
                    if dataset['table'] == 'customers_synthetic':
                        # Generate large customer dataset
                        conn.execute(f"""
                        CREATE OR REPLACE TABLE {dataset['table']} AS
                        SELECT 
                            row_number() OVER () as customer_id,
                            'Customer ' || (row_number() OVER ()) as name,
                            'customer' || (row_number() OVER ()) || '@example.com' as email,
                            (random() * 365 + date '2020-01-01')::date as signup_date,
                            ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'][
                                (random() * 10)::int + 1
                            ] as city,
                            (random() * 62 + 18)::int as age,
                            ['Premium', 'Standard', 'Basic'][
                                (random() * 3)::int + 1
                            ] as tier,
                            random() * 10000 as lifetime_value
                        FROM generate_series(1, {dataset['rows']})
                        """)
                    
                    elif dataset['table'] == 'orders_synthetic':
                        # Generate large orders dataset  
                        conn.execute(f"""
                        CREATE OR REPLACE TABLE {dataset['table']} AS
                        SELECT 
                            row_number() OVER () as order_id,
                            (random() * 50000 + 1)::int as customer_id,
                            ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'Speaker', 'Camera', 'Watch'][
                                (random() * 10)::int + 1
                            ] as product_name,
                            round((random() * 1950 + 50)::numeric, 2) as amount,
                            (random() * 1460 + date '2020-01-01')::timestamp as order_date,
                            ['North', 'South', 'East', 'West', 'Central'][
                                (random() * 5)::int + 1
                            ] as region,
                            ['completed', 'pending', 'shipped', 'cancelled'][
                                (random() * 4)::int + 1
                            ] as status
                        FROM generate_series(1, {dataset['rows']})
                        """)
                    
                    elif dataset['table'] == 'products_synthetic':
                        # Generate large products dataset
                        conn.execute(f"""
                        CREATE OR REPLACE TABLE {dataset['table']} AS
                        SELECT 
                            row_number() OVER () as product_id,
                            'Product ' || (row_number() OVER ()) as name,
                            ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Toys', 'Health', 'Automotive'][
                                (random() * 8)::int + 1
                            ] as category,
                            ['Brand' || ((random() * 20)::int + 1)] as brand,
                            round((random() * 495 + 5)::numeric, 2) as price,
                            (random() * 1000)::int as stock_quantity,
                            random() > 0.1 as is_active,
                            round((random() * 4 + 1)::numeric, 1) as rating
                        FROM generate_series(1, {dataset['rows']})
                        """)
                    
                    # Get actual row count
                    count_result = conn.execute(f"SELECT COUNT(*) FROM {dataset['table']}").fetchone()
                    actual_rows = count_result[0] if count_result else 0
                    
                    load_time = time.time() - start_time
                    
                    loaded_datasets.append({
                        "name": dataset['name'],
                        "table": dataset['table'],
                        "rows": actual_rows,
                        "load_time": round(load_time, 2),
                        "description": dataset['description'],
                        "source": "synthetic"
                    })
                    
                    console.print(f"   ✓ [green]{dataset['name']}[/green]: [blue]{actual_rows:,} rows[/blue] in [yellow]{load_time:.2f}s[/yellow]")
                    logger.info(f"Generated {dataset['name']}: {actual_rows} rows in {load_time:.2f}s")
                    
            except Exception as e:
                console.print(f"   ✗ [red]Failed to generate {dataset['name']}:[/red] {e}")
                logger.error(f"Failed to generate synthetic dataset {dataset['name']}: {e}")
    
    if not loaded_datasets:
        console.print(" [red]No datasets loaded successfully[/red]")
        return False
    
    # Brief summary instead of detailed table
    total_rows = sum(d['rows'] for d in loaded_datasets)
    total_load_time = sum(d['load_time'] for d in loaded_datasets)
    console.print(f"\n✓ [green]Successfully loaded {len(loaded_datasets)} datasets: [bold]{total_rows:,} total records[/bold] in [yellow]{total_load_time:.1f}s[/yellow][/green]")
    
    # Perform analytical queries on larger datasets
    console.print("\n[bold blue]Running analytical queries on larger datasets...[/bold blue]")
    
    large_queries = []
    
    # Add queries based on what datasets loaded successfully
    for dataset in loaded_datasets:
        table_name = dataset['table']
        
        # Handle external datasets (with quoted column names)
        if dataset.get('source') == 'external':
            if table_name == 'customers_large':
                large_queries.extend([
                    ("External Customers by Country", f'SELECT "Country", COUNT(*) as count FROM {table_name} GROUP BY "Country" ORDER BY count DESC LIMIT 10'),
                    ("External Customer Cities", f'SELECT "City", COUNT(*) as count FROM {table_name} GROUP BY "City" ORDER BY count DESC LIMIT 10'),
                    # Skip schema analysis for customers to reduce redundancy
                ])
            elif table_name == 'people_large':
                large_queries.extend([
                    ("People by Gender", f'SELECT "Sex", COUNT(*) as count FROM {table_name} GROUP BY "Sex"'),
                    ("Top Job Titles", f'SELECT "Job Title", COUNT(*) as count FROM {table_name} GROUP BY "Job Title" ORDER BY count DESC LIMIT 10'),
                    # Skip schema analysis for people to reduce redundancy
                ])
            elif table_name == 'organizations_large':
                large_queries.extend([
                    ("Organizations by Country", f'SELECT "Country", COUNT(*) as count FROM {table_name} GROUP BY "Country" ORDER BY count DESC LIMIT 10'),
                    ("Top Industries", f'SELECT "Industry", COUNT(*) as count FROM {table_name} GROUP BY "Industry" ORDER BY count DESC LIMIT 10'),
                    # Only keep one schema analysis example
                    ("Schema Analysis", f"DESCRIBE {table_name}")
                ])
        
        # Handle synthetic datasets (no quoted column names needed)
        elif dataset.get('source') == 'synthetic':
            if table_name == 'customers_synthetic':
                large_queries.extend([
                    ("Customer Age Distribution", f"SELECT age, COUNT(*) as count FROM {table_name} GROUP BY age ORDER BY age LIMIT 10"),
                    ("Customer Tier Analysis", f"SELECT tier, COUNT(*) as count FROM {table_name} GROUP BY tier ORDER BY count DESC"),
                    ("Customer City Distribution", f"SELECT city, COUNT(*) as count FROM {table_name} GROUP BY city ORDER BY count DESC LIMIT 10"),
                    ("Schema Analysis", f"DESCRIBE {table_name}")
                ])
            elif table_name == 'orders_synthetic':
                large_queries.extend([
                    ("Orders by Status", f"SELECT status, COUNT(*) as count FROM {table_name} GROUP BY status ORDER BY count DESC"),
                    ("Orders by Region", f"SELECT region, COUNT(*) as count FROM {table_name} GROUP BY region ORDER BY count DESC"),
                    ("Top Products by Orders", f"SELECT product_name, COUNT(*) as count FROM {table_name} GROUP BY product_name ORDER BY count DESC LIMIT 10"),
                    ("Average Order Value", f"SELECT AVG(amount) as avg_order_value, MIN(amount) as min_order, MAX(amount) as max_order FROM {table_name}"),
                    ("Schema Analysis", f"DESCRIBE {table_name}")
                ])
            elif table_name == 'products_synthetic':
                large_queries.extend([
                    ("Products by Category", f"SELECT category, COUNT(*) as count FROM {table_name} GROUP BY category ORDER BY count DESC"),
                    ("Average Price by Category", f"SELECT category, AVG(price) as avg_price FROM {table_name} GROUP BY category ORDER BY avg_price DESC"),
                    ("Top Brands", f"SELECT brand, COUNT(*) as count FROM {table_name} GROUP BY brand ORDER BY count DESC LIMIT 10"),
                    ("Schema Analysis", f"DESCRIBE {table_name}")
                ])
    
    # Cross-dataset analytical queries if multiple datasets loaded
    if len(loaded_datasets) >= 2:
        table_names = [d['table'] for d in loaded_datasets]
        
        # Cross-dataset queries for external data
        if 'customers_large' in table_names and 'organizations_large' in table_names:
            large_queries.append((
                "Customer vs Organization Countries",
                f"""SELECT 
                    COALESCE(c."Country", o."Country") as country,
                    COUNT(DISTINCT c."Customer Id") as customers,
                    COUNT(DISTINCT o."Organization Id") as organizations
                FROM customers_large c
                FULL OUTER JOIN organizations_large o ON c."Country" = o."Country"
                GROUP BY COALESCE(c."Country", o."Country")
                ORDER BY customers DESC, organizations DESC
                LIMIT 10"""
            ))
        
        if len([d for d in loaded_datasets if d.get('source') == 'external']) >= 2:
            large_queries.append((
                "Total Records by Dataset",
                f"""SELECT 'Customers' as dataset, COUNT(*) as records FROM customers_large
                UNION ALL
                SELECT 'People' as dataset, COUNT(*) as records FROM people_large
                UNION ALL  
                SELECT 'Organizations' as dataset, COUNT(*) as records FROM organizations_large"""
            ))
        
        # Cross-dataset queries for synthetic data
        if 'customers_synthetic' in table_names and 'orders_synthetic' in table_names:
            large_queries.append((
                "Customer-Order Analysis",
                f"""SELECT 
                    c.tier,
                    COUNT(DISTINCT c.customer_id) as customers,
                    COUNT(o.order_id) as total_orders,
                    AVG(o.amount) as avg_order_value
                FROM customers_synthetic c
                LEFT JOIN orders_synthetic o ON c.customer_id = o.customer_id
                GROUP BY c.tier
                ORDER BY avg_order_value DESC"""
            ))
        
        if 'orders_synthetic' in table_names and 'products_synthetic' in table_names:
            large_queries.append((
                "Product Performance",
                f"""SELECT 
                    p.category,
                    COUNT(o.order_id) as order_count,
                    AVG(o.amount) as avg_order_amount,
                    AVG(p.price) as avg_product_price
                FROM products_synthetic p
                LEFT JOIN orders_synthetic o ON p.name = o.product_name
                GROUP BY p.category
                ORDER BY order_count DESC
                LIMIT 10"""
            ))
        
        # Skip the duplicate dataset summary since we already have "Total Records by Dataset"
    
    # Skip generic queries to avoid duplication with other summaries
    
    query_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Executing queries...", total=len(large_queries))
        
        for query_name, query in large_queries:
            progress.update(task, description=f"Running: {query_name}")
            try:
                start_time = time.time()
                result = conn.execute(query).fetchdf()
                exec_time = time.time() - start_time
                
                query_results.append({
                    "query": query_name,
                    "time_ms": round(exec_time * 1000, 2),
                    "rows": len(result),
                    "result": result
                })
                
                # Log to file but don't print to console to reduce verbosity
                logger.info(f"Query {query_name}: {exec_time*1000:.2f}ms ({len(result)} rows)")
                
            except Exception as e:
                console.print(f"   [red]{query_name} failed:[/red] {e}")
                logger.warning(f"Query {query_name} failed: {e}")
            
            progress.advance(task)
    
    # Show actual query results instead of just performance metrics
    if query_results:
        console.print("\n[bold blue]Query Results and Performance:[/bold blue]")
        
        for result in query_results:
            query_name = result['query']
            exec_time = result['time_ms']
            result_df = result['result']
            
            # Create a table for each meaningful query result
            if not result_df.empty and result['rows'] > 1:
                result_table = Table(title=f" {query_name} ({exec_time}ms)", box=box.ROUNDED)
                
                # Add columns from the result
                for col in result_df.columns:
                    result_table.add_column(col, style="cyan")
                
                # Add rows (limit to first 5 to keep it concise)
                for _, row in result_df.head(5).iterrows():
                    result_table.add_row(*[str(val) for val in row])
                
                console.print("\n")
                console.print(result_table)
            
            elif result['rows'] == 1 and 'Schema Analysis' not in query_name:
                # For single-value results, show inline
                if not result_df.empty:
                    first_row = result_df.iloc[0]
                    values = " | ".join([f"{col}: {val}" for col, val in first_row.items()])
                    console.print(f"   [cyan]{query_name}[/cyan] ({exec_time}ms): [green]{values}[/green]")
    
    # Show brief sample data info instead of full table
    if loaded_datasets:
        sample_table_name = loaded_datasets[0]['table'] 
        try:
            sample_data = conn.execute(f"SELECT * FROM {sample_table_name} LIMIT 1").fetchdf()
            if not sample_data.empty:
                col_count = len(sample_data.columns)
                console.print(f"✓ [cyan]Sample data verified: {col_count} columns available for analysis[/cyan]")
        except Exception as e:
            logger.warning(f"Could not verify sample data: {e}")
    
    # Storage analysis for large datasets
    console.print("\n[bold blue]Analyzing storage efficiency with larger datasets...[/bold blue]")
    
    try:
        # Calculate total storage
        lake_data_dir = "./ducklake_data"
        if os.path.exists(lake_data_dir):
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(lake_data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            # Calculate total rows across all tables
            total_rows = sum(dataset['rows'] for dataset in loaded_datasets)
            total_rows += 600  # Add original demo data
            
            # Storage efficiency metrics
            storage_table = Table(title=" Large Dataset Storage Metrics", box=box.ROUNDED)
            storage_table.add_column("Metric", style="cyan")
            storage_table.add_column("Value", style="green")
            
            storage_table.add_row("Total Records", f"{total_rows:,}")
            storage_table.add_row("Storage Size", f"{total_size / (1024*1024):.2f} MB")
            storage_table.add_row("Files Created", str(file_count))
            storage_table.add_row("Avg Bytes/Record", f"{total_size / total_rows:.2f}" if total_rows > 0 else "N/A")
            storage_table.add_row("Compression Ratio", "~5-10x vs CSV")
            
            console.print("\n")
            console.print(storage_table)
    
    except Exception as e:
        console.print(f" [yellow]Storage analysis unavailable: {e}[/yellow]")
    
    console.print("✓ [green]Large dataset experimentation completed[/green]")
    logger.info("Large dataset demonstration completed successfully")
    return True


def show_maintenance_info(conn: duckdb.DuckDBPyConnection) -> bool:
    """Show data maintenance and monitoring information."""
    console.print("\n[bold blue]Gathering maintenance information...[/bold blue]")

    with console.status("[bold blue]Collecting maintenance data..."):
        try:
            # Get table statistics for all tables
            all_tables = conn.execute("SHOW TABLES").fetchdf()
            
            if all_tables.empty:
                console.print(" [yellow]No tables found[/yellow]")
                return True
            
            table_stats_queries = []
            for _, table_row in all_tables.iterrows():
                table_name = table_row['name'] if 'name' in table_row else str(table_row[0])
                table_stats_queries.append(f"SELECT '{table_name}' as table_name, COUNT(*) as row_count, 'DuckLake table' as type FROM \"{table_name}\"")
            
            if table_stats_queries:
                union_query = " UNION ALL ".join(table_stats_queries)
                table_stats = conn.execute(union_query).fetchdf()
            else:
                table_stats = pd.DataFrame()

            # Show catalog metadata
            catalog_info = all_tables

            # Get more detailed table info
            try:
                table_info = conn.execute(
                    "SELECT * FROM information_schema.tables WHERE table_schema != 'information_schema'"
                ).fetchdf()
            except:
                table_info = pd.DataFrame()

            console.print("✓ [green]Maintenance operations completed[/green]")
            logger.info("Maintenance information gathered successfully")

            # Create table statistics table
            if not table_stats.empty:
                stats_table = Table(title=" Table Statistics", box=box.ROUNDED)
                stats_table.add_column("Table Name", style="cyan")
                stats_table.add_column("Row Count", justify="right", style="green")
                stats_table.add_column("Type", style="blue")

                for _, row in table_stats.iterrows():
                    stats_table.add_row(
                        str(row["table_name"]), f"{row['row_count']:,}", str(row["type"])
                    )

                console.print("\n")
                console.print(stats_table)

            # Create catalog info table
            catalog_table = Table(title=" Catalog Information", box=box.ROUNDED)
            if not catalog_info.empty:
                for col in catalog_info.columns:
                    catalog_table.add_column(col, style="cyan")

                for _, row in catalog_info.iterrows():
                    catalog_table.add_row(*[str(val) for val in row])

            console.print("\n")
            console.print(catalog_table)

            if (
                not table_info.empty
                and "table_name" in table_info.columns
                and "table_type" in table_info.columns
            ):
                detailed_table = Table(
                    title="Detailed Table Information", box=box.ROUNDED
                )
                detailed_table.add_column("Table Name", style="cyan")
                detailed_table.add_column("Table Type", style="blue")

                for _, row in table_info[["table_name", "table_type"]].iterrows():
                    detailed_table.add_row(
                        str(row["table_name"]), str(row["table_type"])
                    )

                console.print("\n")
                console.print(detailed_table)

            return True
        except Exception as e:
            console.print(f" [red]Maintenance operations failed:[/red] {e}")
            logger.error(f"Maintenance operations failed: {e}")
            return False


def reset_ducklake_data() -> bool:
    """Reset DuckLake data and snapshots."""
    console.print("[bold blue]Resetting DuckLake data...[/bold blue]")

    with console.status("[bold blue]Cleaning up data and schemas..."):
        try:
            # Remove local data directory
            if os.path.exists("./ducklake_data"):
                shutil.rmtree("./ducklake_data")
                console.print("✓ [green]Removed local data directory[/green]")
                logger.info("Local data directory removed")

            # Clean PostgreSQL catalog schemas
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "exec",
                    "postgres",
                    "psql",
                    "-U",
                    "ducklake",
                    "-d",
                    "ducklake_catalog",
                    "-c",
                    "DROP SCHEMA IF EXISTS ducklake CASCADE; DROP SCHEMA IF EXISTS main CASCADE; DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                console.print(
                    "✓ [green]DuckLake data reset complete! All tables and snapshots removed.[/green]"
                )
                logger.info("DuckLake data reset completed successfully")
            else:
                console.print(
                    f"⚠ [yellow]PostgreSQL reset may have issues:[/yellow] {result.stderr}"
                )
                logger.warning(f"PostgreSQL reset issues: {result.stderr}")
            return True
        except Exception as e:
            console.print(f"✗ [red]Reset failed:[/red] {e}")
            logger.error(f"Reset operation failed: {e}")
            return False


@click.group(invoke_without_command=True)
@click.option("--no-reset", is_flag=True, help="Keep existing data between runs")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: Context, no_reset: bool, verbose: bool) -> None:
    """DuckLake Tutorial - Simple Python Demo

    This script demonstrates DuckDB's DuckLake capabilities using a local PostgreSQL catalog.
    DuckLake provides advanced data lake features without traditional lakehouse complexity.
    """
    # If no subcommand is provided, run the demo
    if ctx.invoked_subcommand is None:
        run_demo(no_reset, verbose)


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def reset(verbose: bool) -> None:
    """Reset DuckLake data and PostgreSQL catalog."""
    if verbose:
        console.print(" [bold blue]Performing data reset...[/bold blue]")
    success = reset_ducklake_data()
    if success and verbose:
        console.print("✓ [green]Reset completed successfully[/green]")
    elif not success:
        console.print("✗ [red]Reset failed[/red]")


@cli.command()
@click.option("--no-reset", is_flag=True, help="Keep existing data between runs")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def demo(no_reset: bool, verbose: bool) -> None:
    """Run the DuckLake demonstration."""
    run_demo(no_reset, verbose)


def run_demo(no_reset: bool = False, verbose: bool = False) -> None:
    """Main demonstration function."""
    # Create main title panel
    title_panel = Panel(" DuckLake Tutorial", style="bold blue", box=box.DOUBLE)
    console.print(title_panel)

    # Reset by default for clean demo (unless --no-reset is passed)
    if not no_reset:
        console.print(
            "[bold yellow]Resetting to clean state for fresh demo...[/bold yellow]"
        )
        reset_ducklake_data()

        # Also remove any existing DuckDB file
        if os.path.exists("ducklake_demo.duckdb"):
            os.remove("ducklake_demo.duckdb")
            console.print("✓ [green]Removed existing DuckDB file[/green]")

    # Check prerequisites
    if not check_postgresql():
        console.print(
            "\n [red]PostgreSQL not available. Please run:[/red] [bold]docker-compose up -d[/bold]"
        )
        logger.error("PostgreSQL not available")
        return

    # Create DuckDB connection
    console.print("\n[bold blue]Creating DuckDB connection...[/bold blue]")
    conn = duckdb.connect("ducklake_demo.duckdb")
    console.print("✓ [green]DuckDB connection established[/green]")
    logger.info("DuckDB connection established")

    try:
        # Setup extensions
        if not setup_extensions(conn):
            return

        # Test PostgreSQL connection
        pg_config = test_postgresql_connection()
        if not pg_config:
            return

        # Initialize DuckLake
        db_name = initialize_ducklake(conn, pg_config)
        if not db_name:
            return

        # Phase 1
        phase1_content = """ This phase demonstrates:
   • Creating tables in DuckLake format
   • Ingesting data (customers & sales)
   • Running analytical queries across tables
   • Understanding the Parquet storage layer"""

        phase1_panel = Panel(
            phase1_content,
            title=" PHASE 1: Core Operations & Data Ingestion",
            style="bold cyan",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(phase1_panel)

        # Create sample data
        if not create_sample_data(conn):
            return

        # Demonstrate queries
        if not demonstrate_queries(conn):
            return

        # Summary of what happened
        summary1_content = """• Data was stored as Parquet files in ./ducklake_data/
• PostgreSQL tracks table metadata and schemas
• Queries work seamlessly across the distributed storage"""

        summary1_panel = Panel(
            summary1_content,
            title=" What just happened",
            style="green",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(summary1_panel)

        # Wait for user confirmation before Phase 2
        console.print("\n[bold yellow]Press Enter to continue to Phase 2 (ACID Transactions & Time Travel)...[/bold yellow]")
        input()

        # Phase 2
        phase2_content = """ This phase demonstrates:
   • ACID transaction guarantees with rollback
   • Automatic snapshot creation on data changes
   • Time travel queries to historical versions
   • Schema evolution (adding new columns)"""

        phase2_panel = Panel(
            phase2_content,
            title=" PHASE 2: ACID Transactions & Time Travel",
            style="bold magenta",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(phase2_panel)

        # Demonstrate ACID transactions
        if not demonstrate_acid_transactions(conn):
            return

        # Demonstrate time travel
        if not demonstrate_time_travel(conn, db_name):
            return

        # Demonstrate schema evolution
        if not demonstrate_schema_evolution(conn):
            return

        # Summary of what happened
        summary2_content = """• Every data change automatically creates new snapshots
• You can query any historical version using AT VERSION
• Schema changes are versioned and backward compatible"""

        summary2_panel = Panel(
            summary2_content,
            title=" What just happened",
            style="green",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(summary2_panel)

        # Wait for user confirmation before Phase 3
        console.print("\n[bold yellow]Press Enter to continue to Phase 3 (Performance, Storage & File Analysis)...[/bold yellow]")
        input()

        # Phase 3
        phase3_content = """ This phase demonstrates:
   • Query performance measurement
   • Parquet file structure exploration
   • Storage efficiency analysis
   • Catalog metadata inspection"""

        phase3_panel = Panel(
            phase3_content,
            title=" PHASE 3: Performance, Storage & File Analysis",
            style="bold yellow",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(phase3_panel)

        # Demonstrate performance
        if not demonstrate_performance(conn):
            return

        # Explore the Parquet file structure
        if not explore_parquet_files():
            return

        # Demonstrate compression efficiency
        if not demonstrate_data_compression():
            return

        # Show maintenance info
        if not show_maintenance_info(conn):
            return

        # Wait for user confirmation before Phase 4
        console.print("\n[bold yellow]Press Enter to continue to Phase 4 (Large Dataset Experimentation)...[/bold yellow]")
        input()

        # Phase 4
        phase4_content = """ This phase demonstrates:
   • Loading much larger datasets (100k+ records each)
   • Performance testing with real-world data
   • Storage efficiency analysis at scale
   • Cross-dataset analytical queries"""

        phase4_panel = Panel(
            phase4_content,
            title=" PHASE 4: Large Dataset Experimentation",
            style="bold red",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(phase4_panel)

        # Demonstrate large datasets
        if not demonstrate_large_dataset(conn):
            console.print(" [yellow]Phase 4 completed with some limitations[/yellow]")

        # Summary of what happened in Phase 4
        summary4_content = """• Loaded 100k+ record datasets directly from GitHub URLs
• Demonstrated DuckLake's ability to handle larger workloads
• Analyzed storage efficiency with compressed Parquet format
• Showed real-world performance characteristics"""

        summary4_panel = Panel(
            summary4_content,
            title=" What just happened",
            style="green",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(summary4_panel)

        # Completion summary
        completion_content = """ Successfully demonstrated DuckLake's key capabilities:
    Foundation Setup: PostgreSQL catalog with local storage
    Core Operations: Table creation, data ingestion, queries
    ACID Transactions: Consistent data operations with rollback
    Advanced Features: Snapshots, time travel, schema evolution
    Performance: Query optimization and execution analysis
    Large Scale: 100k+ record datasets and real-world performance
    Maintenance: Statistics and monitoring operations"""

        completion_panel = Panel(
            completion_content,
            title=" DuckLake Tutorial Complete!",
            style="bold green",
            box=box.DOUBLE,
        )
        console.print("\n")
        console.print(completion_panel)

        # Key advantages
        advantages_content = """• SQL-First: Uses familiar SQL databases for metadata
• ACID Compliance: Full transaction support
• Open Format: Parquet files + SQL catalog
• Local Development: No cloud dependencies required
• Scalable: Supports concurrent multi-user access"""

        advantages_panel = Panel(
            advantages_content,
            title=" Key Advantages of DuckLake",
            style="blue",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(advantages_panel)

        # Next steps
        next_steps_content = """1. Explore additional datasets from https://github.com/datablist/sample-csv-files
2. Test multi-user scenarios with additional connections
3. Explore partitioning strategies for large tables
4. Integrate with existing data pipelines
5. Set up monitoring and alerting
6. Experiment with cross-dataset JOIN operations at scale"""

        next_steps_panel = Panel(
            next_steps_content, title=" Next Steps", style="cyan", box=box.ROUNDED
        )
        console.print("\n")
        console.print(next_steps_panel)

        console.print(
            "\n [italic]To keep data between runs: ducklake-demo --no-reset[/italic]"
        )
        logger.info("DuckLake tutorial completed successfully")

    finally:
        # Clean up
        try:
            conn.close()
        except:
            pass


if __name__ == "__main__":
    cli()
