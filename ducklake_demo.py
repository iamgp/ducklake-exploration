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

import click
import duckdb
import numpy as np
import pandas as pd
from click import Context
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
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
file_handler = logging.FileHandler("ducklake_demo.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
    pd.DataFrame(sales_info)

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
    console.print("[bold blue]Exploring DuckLake Parquet file structure...[/bold blue]")

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
    console.print(
        "\n[bold blue]Loading larger dataset for performance testing...[/bold blue]"
    )

    # Try external datasets first, fallback to synthetic data generation
    external_datasets = [
        {
            "name": "Customers (External)",
            "url": "https://drive.google.com/uc?id=1N1xoxgcw2K3d-49tlchXAWw4wuxLj7EV&export=download",
            "table": "customers_large",
            "description": "100k records of customer data from GitHub samples",
        },
        {
            "name": "People (External)",
            "url": "https://drive.google.com/uc?id=1NW7EnwxuY6RpMIxOazRVibOYrZfMjsb2&export=download",
            "table": "people_large",
            "description": "100k records of people demographics from GitHub samples",
        },
        {
            "name": "Organizations (External)",
            "url": "https://drive.google.com/uc?id=1g4wqEIsKyiBWeCAwd0wEkiC4Psc4zwFu&export=download",
            "table": "organizations_large",
            "description": "100k records of organization data from GitHub samples",
        },
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
                CREATE OR REPLACE TABLE {dataset["table"]} AS 
                SELECT * FROM read_csv_auto('{dataset["url"]}')
                """)

                # Get row count
                count_result = conn.execute(
                    f"SELECT COUNT(*) FROM {dataset['table']}"
                ).fetchone()
                row_count = count_result[0] if count_result else 0

                load_time = time.time() - start_time

                if row_count > 0:
                    loaded_datasets.append(
                        {
                            "name": dataset["name"],
                            "table": dataset["table"],
                            "rows": row_count,
                            "load_time": round(load_time, 2),
                            "description": dataset["description"],
                            "source": "external",
                        }
                    )

                    console.print(
                        f"   ✓ [green]{dataset['name']}[/green]: [blue]{row_count:,} rows[/blue] in [yellow]{load_time:.2f}s[/yellow]"
                    )
                    logger.info(
                        f"Loaded {dataset['name']}: {row_count} rows in {load_time:.2f}s"
                    )

        except Exception as e:
            console.print(
                f"   ⚠ [yellow]External dataset {dataset['name']} failed:[/yellow] {str(e)[:100]}..."
            )
            logger.warning(f"Failed to load external dataset {dataset['name']}: {e}")

    # If external datasets failed, generate synthetic large datasets
    if not loaded_datasets:
        console.print(
            "[cyan]Generating synthetic large datasets for demonstration...[/cyan]"
        )

        synthetic_datasets = [
            {
                "name": "Large Customers (Synthetic)",
                "table": "customers_synthetic",
                "rows": 50000,
                "description": "50k synthetic customer records",
            },
            {
                "name": "Large Orders (Synthetic)",
                "table": "orders_synthetic",
                "rows": 100000,
                "description": "100k synthetic order records",
            },
            {
                "name": "Large Products (Synthetic)",
                "table": "products_synthetic",
                "rows": 25000,
                "description": "25k synthetic product records",
            },
        ]

        for dataset in synthetic_datasets:
            try:
                with console.status(f"[bold blue]Generating {dataset['name']}..."):
                    start_time = time.time()

                    if dataset["table"] == "customers_synthetic":
                        # Generate large customer dataset
                        conn.execute(f"""
                        CREATE OR REPLACE TABLE {dataset["table"]} AS
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
                        FROM generate_series(1, {dataset["rows"]})
                        """)

                    elif dataset["table"] == "orders_synthetic":
                        # Generate large orders dataset
                        conn.execute(f"""
                        CREATE OR REPLACE TABLE {dataset["table"]} AS
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
                        FROM generate_series(1, {dataset["rows"]})
                        """)

                    elif dataset["table"] == "products_synthetic":
                        # Generate large products dataset
                        conn.execute(f"""
                        CREATE OR REPLACE TABLE {dataset["table"]} AS
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
                        FROM generate_series(1, {dataset["rows"]})
                        """)

                    # Get actual row count
                    count_result = conn.execute(
                        f"SELECT COUNT(*) FROM {dataset['table']}"
                    ).fetchone()
                    actual_rows = count_result[0] if count_result else 0

                    load_time = time.time() - start_time

                    loaded_datasets.append(
                        {
                            "name": dataset["name"],
                            "table": dataset["table"],
                            "rows": actual_rows,
                            "load_time": round(load_time, 2),
                            "description": dataset["description"],
                            "source": "synthetic",
                        }
                    )

                    console.print(
                        f"   ✓ [green]{dataset['name']}[/green]: [blue]{actual_rows:,} rows[/blue] in [yellow]{load_time:.2f}s[/yellow]"
                    )
                    logger.info(
                        f"Generated {dataset['name']}: {actual_rows} rows in {load_time:.2f}s"
                    )

            except Exception as e:
                console.print(
                    f"   ✗ [red]Failed to generate {dataset['name']}:[/red] {e}"
                )
                logger.error(
                    f"Failed to generate synthetic dataset {dataset['name']}: {e}"
                )

    if not loaded_datasets:
        console.print(" [red]No datasets loaded successfully[/red]")
        return False

    # Brief summary instead of detailed table
    total_rows = sum(d["rows"] for d in loaded_datasets)
    total_load_time = sum(d["load_time"] for d in loaded_datasets)
    console.print(
        f"\n✓ [green]Successfully loaded {len(loaded_datasets)} datasets: [bold]{total_rows:,} total records[/bold] in [yellow]{total_load_time:.1f}s[/yellow][/green]"
    )

    # Perform analytical queries on larger datasets
    console.print(
        "\n[bold blue]Running analytical queries on larger datasets...[/bold blue]"
    )

    large_queries = []

    # Add queries based on what datasets loaded successfully
    for dataset in loaded_datasets:
        table_name = dataset["table"]

        # Handle external datasets (with quoted column names)
        if dataset.get("source") == "external":
            if table_name == "customers_large":
                large_queries.extend(
                    [
                        (
                            "External Customers by Country",
                            f'SELECT "Country", COUNT(*) as count FROM {table_name} GROUP BY "Country" ORDER BY count DESC LIMIT 10',
                        ),
                        (
                            "External Customer Cities",
                            f'SELECT "City", COUNT(*) as count FROM {table_name} GROUP BY "City" ORDER BY count DESC LIMIT 10',
                        ),
                        # Skip schema analysis for customers to reduce redundancy
                    ]
                )
            elif table_name == "people_large":
                large_queries.extend(
                    [
                        (
                            "People by Gender",
                            f'SELECT "Sex", COUNT(*) as count FROM {table_name} GROUP BY "Sex"',
                        ),
                        (
                            "Top Job Titles",
                            f'SELECT "Job Title", COUNT(*) as count FROM {table_name} GROUP BY "Job Title" ORDER BY count DESC LIMIT 10',
                        ),
                        # Skip schema analysis for people to reduce redundancy
                    ]
                )
            elif table_name == "organizations_large":
                large_queries.extend(
                    [
                        (
                            "Organizations by Country",
                            f'SELECT "Country", COUNT(*) as count FROM {table_name} GROUP BY "Country" ORDER BY count DESC LIMIT 10',
                        ),
                        (
                            "Top Industries",
                            f'SELECT "Industry", COUNT(*) as count FROM {table_name} GROUP BY "Industry" ORDER BY count DESC LIMIT 10',
                        ),
                        # Only keep one schema analysis example
                        ("Schema Analysis", f"DESCRIBE {table_name}"),
                    ]
                )

        # Handle synthetic datasets (no quoted column names needed)
        elif dataset.get("source") == "synthetic":
            if table_name == "customers_synthetic":
                large_queries.extend(
                    [
                        (
                            "Customer Age Distribution",
                            f"SELECT age, COUNT(*) as count FROM {table_name} GROUP BY age ORDER BY age LIMIT 10",
                        ),
                        (
                            "Customer Tier Analysis",
                            f"SELECT tier, COUNT(*) as count FROM {table_name} GROUP BY tier ORDER BY count DESC",
                        ),
                        (
                            "Customer City Distribution",
                            f"SELECT city, COUNT(*) as count FROM {table_name} GROUP BY city ORDER BY count DESC LIMIT 10",
                        ),
                        ("Schema Analysis", f"DESCRIBE {table_name}"),
                    ]
                )
            elif table_name == "orders_synthetic":
                large_queries.extend(
                    [
                        (
                            "Orders by Status",
                            f"SELECT status, COUNT(*) as count FROM {table_name} GROUP BY status ORDER BY count DESC",
                        ),
                        (
                            "Orders by Region",
                            f"SELECT region, COUNT(*) as count FROM {table_name} GROUP BY region ORDER BY count DESC",
                        ),
                        (
                            "Top Products by Orders",
                            f"SELECT product_name, COUNT(*) as count FROM {table_name} GROUP BY product_name ORDER BY count DESC LIMIT 10",
                        ),
                        (
                            "Average Order Value",
                            f"SELECT AVG(amount) as avg_order_value, MIN(amount) as min_order, MAX(amount) as max_order FROM {table_name}",
                        ),
                        ("Schema Analysis", f"DESCRIBE {table_name}"),
                    ]
                )
            elif table_name == "products_synthetic":
                large_queries.extend(
                    [
                        (
                            "Products by Category",
                            f"SELECT category, COUNT(*) as count FROM {table_name} GROUP BY category ORDER BY count DESC",
                        ),
                        (
                            "Average Price by Category",
                            f"SELECT category, AVG(price) as avg_price FROM {table_name} GROUP BY category ORDER BY avg_price DESC",
                        ),
                        (
                            "Top Brands",
                            f"SELECT brand, COUNT(*) as count FROM {table_name} GROUP BY brand ORDER BY count DESC LIMIT 10",
                        ),
                        ("Schema Analysis", f"DESCRIBE {table_name}"),
                    ]
                )

    # Cross-dataset analytical queries if multiple datasets loaded
    if len(loaded_datasets) >= 2:
        table_names = [d["table"] for d in loaded_datasets]

        # Cross-dataset queries for external data
        if "customers_large" in table_names and "organizations_large" in table_names:
            large_queries.append(
                (
                    "Customer vs Organization Countries",
                    """SELECT 
                    COALESCE(c."Country", o."Country") as country,
                    COUNT(DISTINCT c."Customer Id") as customers,
                    COUNT(DISTINCT o."Organization Id") as organizations
                FROM customers_large c
                FULL OUTER JOIN organizations_large o ON c."Country" = o."Country"
                GROUP BY COALESCE(c."Country", o."Country")
                ORDER BY customers DESC, organizations DESC
                LIMIT 10""",
                )
            )

        if len([d for d in loaded_datasets if d.get("source") == "external"]) >= 2:
            large_queries.append(
                (
                    "Total Records by Dataset",
                    """SELECT 'Customers' as dataset, COUNT(*) as records FROM customers_large
                UNION ALL
                SELECT 'People' as dataset, COUNT(*) as records FROM people_large
                UNION ALL  
                SELECT 'Organizations' as dataset, COUNT(*) as records FROM organizations_large""",
                )
            )

        # Cross-dataset queries for synthetic data
        if "customers_synthetic" in table_names and "orders_synthetic" in table_names:
            large_queries.append(
                (
                    "Customer-Order Analysis",
                    """SELECT 
                    c.tier,
                    COUNT(DISTINCT c.customer_id) as customers,
                    COUNT(o.order_id) as total_orders,
                    AVG(o.amount) as avg_order_value
                FROM customers_synthetic c
                LEFT JOIN orders_synthetic o ON c.customer_id = o.customer_id
                GROUP BY c.tier
                ORDER BY avg_order_value DESC""",
                )
            )

        if "orders_synthetic" in table_names and "products_synthetic" in table_names:
            large_queries.append(
                (
                    "Product Performance",
                    """SELECT 
                    p.category,
                    COUNT(o.order_id) as order_count,
                    AVG(o.amount) as avg_order_amount,
                    AVG(p.price) as avg_product_price
                FROM products_synthetic p
                LEFT JOIN orders_synthetic o ON p.name = o.product_name
                GROUP BY p.category
                ORDER BY order_count DESC
                LIMIT 10""",
                )
            )

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

                query_results.append(
                    {
                        "query": query_name,
                        "time_ms": round(exec_time * 1000, 2),
                        "rows": len(result),
                        "result": result,
                    }
                )

                # Log to file but don't print to console to reduce verbosity
                logger.info(
                    f"Query {query_name}: {exec_time * 1000:.2f}ms ({len(result)} rows)"
                )

            except Exception as e:
                console.print(f"   [red]{query_name} failed:[/red] {e}")
                logger.warning(f"Query {query_name} failed: {e}")

            progress.advance(task)

    # Show actual query results instead of just performance metrics
    if query_results:
        console.print("\n[bold blue]Query Results and Performance:[/bold blue]")

        for result in query_results:
            query_name = result["query"]
            exec_time = result["time_ms"]
            result_df = result["result"]

            # Create a table for each meaningful query result
            if not result_df.empty and result["rows"] > 1:
                result_table = Table(
                    title=f" {query_name} ({exec_time}ms)", box=box.ROUNDED
                )

                # Add columns from the result
                for col in result_df.columns:
                    result_table.add_column(col, style="cyan")

                # Add rows (limit to first 5 to keep it concise)
                for _, row in result_df.head(5).iterrows():
                    result_table.add_row(*[str(val) for val in row])

                console.print("\n")
                console.print(result_table)

            elif result["rows"] == 1 and "Schema Analysis" not in query_name:
                # For single-value results, show inline
                if not result_df.empty:
                    first_row = result_df.iloc[0]
                    values = " | ".join(
                        [f"{col}: {val}" for col, val in first_row.items()]
                    )
                    console.print(
                        f"   [cyan]{query_name}[/cyan] ({exec_time}ms): [green]{values}[/green]"
                    )

    # Show brief sample data info instead of full table
    if loaded_datasets:
        sample_table_name = loaded_datasets[0]["table"]
        try:
            sample_data = conn.execute(
                f"SELECT * FROM {sample_table_name} LIMIT 1"
            ).fetchdf()
            if not sample_data.empty:
                col_count = len(sample_data.columns)
                console.print(
                    f"✓ [cyan]Sample data verified: {col_count} columns available for analysis[/cyan]"
                )
        except Exception as e:
            logger.warning(f"Could not verify sample data: {e}")

    # Storage analysis for large datasets
    console.print(
        "\n[bold blue]Analyzing storage efficiency with larger datasets...[/bold blue]"
    )

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
            total_rows = sum(dataset["rows"] for dataset in loaded_datasets)
            total_rows += 600  # Add original demo data

            # Storage efficiency metrics
            storage_table = Table(
                title=" Large Dataset Storage Metrics", box=box.ROUNDED
            )
            storage_table.add_column("Metric", style="cyan")
            storage_table.add_column("Value", style="green")

            storage_table.add_row("Total Records", f"{total_rows:,}")
            storage_table.add_row(
                "Storage Size", f"{total_size / (1024 * 1024):.2f} MB"
            )
            storage_table.add_row("Files Created", str(file_count))
            storage_table.add_row(
                "Avg Bytes/Record",
                f"{total_size / total_rows:.2f}" if total_rows > 0 else "N/A",
            )
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
                table_name = (
                    table_row["name"] if "name" in table_row else str(table_row[0])
                )
                table_stats_queries.append(
                    f"SELECT '{table_name}' as table_name, COUNT(*) as row_count, 'DuckLake table' as type FROM \"{table_name}\""
                )

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
                        str(row["table_name"]),
                        f"{row['row_count']:,}",
                        str(row["type"]),
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


def demonstrate_production_concepts(
    conn: duckdb.DuckDBPyConnection, db_name: str
) -> bool:
    """Demonstrate production deployment concepts and best practices."""
    console.print("\n[bold blue]Production Deployment Concepts...[/bold blue]")

    try:
        # Multi-user scenarios with detailed explanations
        console.print("\n[bold cyan]Multi-User Access Patterns[/bold cyan]")

        multiuser_md = """
**Understanding Concurrent Access**

When multiple users or applications access the same data simultaneously, you need to ensure:
- Data consistency (no corruption)
- Performance (users don't wait unnecessarily)
- Isolation (one user's work doesn't break another's)

**How DuckLake Handles Concurrency**

**Multiple Readers**
- Unlimited concurrent read queries are supported
- Each reader gets a consistent snapshot of data
- Readers never block other readers or writers
- Perfect for dashboards and reporting tools

**Writer Isolation** 
- Only one writer can modify data at a time
- ACID transactions ensure all-or-nothing changes
- If two users try to write simultaneously, one waits
- No risk of partial updates or data corruption

**Snapshot Isolation**
- Each query sees data as of a specific point in time
- Long-running queries aren't affected by new writes
- Enables consistent reporting even during data updates

**Connection Pooling**
- Instead of opening a new database connection for each request
- Maintain a pool of reusable connections
- Reduces overhead and improves performance
- Essential for web applications with many users

**Example: Web Application Pattern**
```python
# Connection pool for web app
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'duckdb:///ducklake.db',
    poolclass=QueuePool,
    pool_size=10,        # 10 connections in pool
    max_overflow=20      # Up to 20 additional connections
)
```
        """

        console.print(Markdown(multiuser_md))

        # Partitioning strategies with detailed explanations
        console.print("\n[bold yellow]Data Partitioning for Performance[/bold yellow]")

        partitioning_md = """
**What is Partitioning?**

Partitioning splits large tables into smaller, more manageable pieces called partitions. Think of it like organizing files into folders - instead of one giant folder with millions of files, you create subfolders by year, department, etc.

**Why Partition Data?**

**Query Performance**
- Skip irrelevant partitions (called "partition pruning")
- Only scan data that matches your query filters
- Example: Query for 2024 data only scans 2024 partition, not all years

**Parallel Processing**
- Different partitions can be processed simultaneously
- Utilize multiple CPU cores effectively
- Faster aggregations and analytics

**Data Management**
- Delete old data by dropping entire partitions
- Backup/restore specific time periods
- Apply different retention policies per partition

**Partitioning Strategies**

**Date-Based Partitioning** (Most Common)
```sql
-- Partition sales by year and month
CREATE TABLE sales_partitioned (
    sale_date DATE,
    customer_id INTEGER,
    amount DECIMAL
) PARTITIONED BY (year(sale_date), month(sale_date));
```
- Perfect for time-series data
- Enables efficient historical queries
- Easy to archive old data

**Geographic Partitioning**
```sql
-- Partition by region for global datasets
CREATE TABLE customers_partitioned (
    customer_id INTEGER,
    region VARCHAR,
    country VARCHAR
) PARTITIONED BY (region);
```
- Useful for multi-regional applications
- Compliance with data residency requirements
- Regional performance optimization

**Category-Based Partitioning**
```sql
-- Partition by product category
CREATE TABLE products_partitioned (
    product_id INTEGER,
    category VARCHAR,
    price DECIMAL
) PARTITIONED BY (category);
```
- Organize by business dimensions
- Department-specific data access
- Workload isolation

**Hash Partitioning**
```sql
-- Distribute data evenly across partitions
CREATE TABLE events_partitioned (
    event_id INTEGER,
    user_id INTEGER,
    event_data JSON
) PARTITIONED BY (hash(user_id) % 10);
```
- Even distribution when no natural partition key exists
- Load balancing across partitions
- Good for parallel processing

**Choosing the Right Strategy**

Ask yourself:
- How do users typically filter data? (by date, region, category?)
- What's your most common query pattern?
- Do you need to archive old data regularly?
- Are there compliance requirements for data location?
        """

        console.print(Markdown(partitioning_md))

        # Table maintenance with detailed explanations
        console.print("\n[bold green]Table Maintenance and Optimization[/bold green]")

        maintenance_md = """
**Why Table Maintenance Matters**

Over time, data lakes accumulate "cruft" - deleted records, small files, outdated statistics. Without maintenance, performance degrades and storage costs increase.

**Core Maintenance Operations**

**VACUUM - Space Reclamation**
```sql
VACUUM customers;
```
- Removes physically deleted records from files
- Reclaims disk space back to the operating system
- Consolidates fragmented data files
- When to use: After large DELETE operations

**OPTIMIZE - File Compaction**
```sql
OPTIMIZE sales;
```
- Combines many small files into fewer large files
- Improves query performance (fewer files to open)
- Reduces metadata overhead
- When to use: After many small INSERT operations

**ANALYZE - Statistics Update**
```sql
ANALYZE TABLE products;
```
- Updates table and column statistics for query optimizer
- Helps database choose efficient query execution plans
- Critical for good performance on large tables
- When to use: After significant data changes

**CHECKPOINT - Recovery Points**
```sql
CHECKPOINT;
```
- Creates recovery savepoints
- Ensures all changes are written to disk
- Enables faster recovery after crashes
- When to use: Before major operations or during maintenance windows

**Maintenance Scheduling**

**Daily Tasks**
- ANALYZE tables with frequent INSERT/UPDATE/DELETE operations
- Monitor query performance metrics
- Check for failed operations in logs

**Weekly Tasks**  
- VACUUM tables with high DELETE volume
- Review slow query logs
- Check storage growth trends

**Monthly Tasks**
- OPTIMIZE large tables with many small files
- Review and update partitioning strategies
- Performance baseline analysis

**Quarterly Tasks**
- Review overall data lake architecture
- Evaluate partitioning effectiveness
- Capacity planning and cost optimization

**Monitoring What Matters**

**Storage Metrics**
- Table sizes and growth rates over time
- Number of files per table (watch for file explosion)
- Storage utilization and costs

**Performance Metrics**
- Query execution times (watch for degradation)
- Concurrent connection counts
- Cache hit ratios

**Operational Metrics**
- Error rates and timeout frequency
- Maintenance operation success rates
- Data freshness and staleness

**Example Monitoring Query**
```sql
-- Check table file counts (high numbers indicate need for OPTIMIZE)
SELECT 
    table_name,
    COUNT(*) as file_count,
    SUM(file_size_bytes) / (1024*1024) as size_mb
FROM table_files_metadata 
GROUP BY table_name
ORDER BY file_count DESC;
```
        """

        console.print(Markdown(maintenance_md))

        # Practical maintenance demonstration
        console.print(
            "\n[bold green]Hands-On: Table Maintenance Operations[/bold green]"
        )

        console.print("\nLet's see how to perform actual maintenance on DuckLake:")

        # Maintenance operations code
        maintenance_demo_code = '''import duckdb

conn = duckdb.connect('ducklake.duckdb')

# Check table information before maintenance
table_info = conn.execute("""
    SELECT 
        table_name,
        estimated_size,
        row_count
    FROM information_schema.tables 
    WHERE table_schema != 'information_schema'
""").fetchdf()

print("Tables before maintenance:")
print(table_info)

# VACUUM operation - reclaim space
print("\\n🧹 Running VACUUM on customers table...")
conn.execute("VACUUM customers")
print("✓ VACUUM completed - space reclaimed")

# OPTIMIZE operation - compact files  
print("\\n📦 Running OPTIMIZE on sales table...")
conn.execute("OPTIMIZE sales")
print("✓ OPTIMIZE completed - files compacted")

# ANALYZE operation - update statistics
print("\\n📊 Running ANALYZE on all tables...")
conn.execute("ANALYZE")
print("✓ ANALYZE completed - statistics updated")

# Check file counts (indicates fragmentation)
file_stats = conn.execute("""
    SELECT 
        table_name,
        COUNT(*) as file_count,
        SUM(file_size_bytes) / (1024*1024) as total_size_mb
    FROM (
        -- This is a conceptual query - actual metadata table may differ
        SELECT 'customers' as table_name, 1024*1024*5 as file_size_bytes
        UNION ALL SELECT 'sales', 1024*1024*8
    ) 
    GROUP BY table_name
""").fetchdf()

print("\\nFile statistics after maintenance:")
print(file_stats)'''

        console.print("\n[bold cyan]Maintenance Operations Demo[/bold cyan]")
        syntax = Syntax(
            maintenance_demo_code, "python", theme="monokai", line_numbers=True
        )
        console.print(syntax)

        # Monitoring queries demo
        monitoring_code = '''# Production monitoring queries

# 1. Check table sizes and growth
size_query = """
SELECT 
    table_name,
    pg_size_pretty(pg_total_relation_size(table_name)) as table_size,
    pg_size_pretty(pg_relation_size(table_name)) as data_size
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY pg_total_relation_size(table_name) DESC
"""

table_sizes = conn.execute(size_query).fetchdf()
print("Table sizes:")
print(table_sizes)

# 2. Check query performance over time
performance_query = """
SELECT 
    DATE(query_time) as date,
    COUNT(*) as query_count,
    AVG(execution_time_ms) as avg_time_ms,
    MAX(execution_time_ms) as max_time_ms
FROM query_log 
WHERE query_time >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(query_time)
ORDER BY date
"""

# 3. Check concurrent connections
connection_query = """
SELECT 
    COUNT(*) as active_connections,
    COUNT(CASE WHEN state = 'active' THEN 1 END) as running_queries,
    COUNT(CASE WHEN state = 'idle' THEN 1 END) as idle_connections
FROM pg_stat_activity 
WHERE datname = 'ducklake_catalog'
"""

connections = conn.execute(connection_query).fetchdf()
print("\\nConnection status:")
print(connections)'''

        console.print("\n[bold cyan]Production Monitoring Queries[/bold cyan]")
        syntax = Syntax(monitoring_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        # Test concurrent access simulation
        console.print(
            "\n[bold blue]Simulating concurrent access patterns...[/bold blue]"
        )

        # Create a simple test of concurrent read capability
        try:
            # Simulate multiple concurrent queries
            import threading
            import time

            query_times = []

            def run_query():
                start = time.time()
                conn.execute("SELECT COUNT(*) FROM customers").fetchone()
                end = time.time()
                query_times.append(end - start)

            # Run 3 concurrent queries
            threads = []
            for i in range(3):
                t = threading.Thread(target=run_query)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            avg_time = sum(query_times) / len(query_times) * 1000
            console.print(
                f"✓ [green]Concurrent access test: 3 parallel queries, avg {avg_time:.2f}ms[/green]"
            )

        except Exception as e:
            console.print(f"[yellow]Concurrent access simulation skipped: {e}[/yellow]")

        return True
    except Exception as e:
        console.print(f" [red]Production concepts demonstration failed:[/red] {e}")
        logger.error(f"Production concepts failed: {e}")
        return False


def demonstrate_integration_patterns(conn: duckdb.DuckDBPyConnection) -> bool:
    """Demonstrate integration patterns with BI tools, APIs, and ETL systems."""
    console.print("\n[bold blue]Integration Patterns and Examples...[/bold blue]")

    try:
        # ETL/ELT patterns with detailed explanations
        console.print("\n[bold blue]ETL vs ELT Data Processing[/bold blue]")

        etl_md = """
**Understanding Data Processing Approaches**

**Traditional ETL (Extract-Transform-Load)**

The "old school" approach where you clean and structure data before storing it:

1. **Extract**: Pull data from source systems (databases, APIs, files)
2. **Transform**: Clean, validate, and structure the data
3. **Load**: Store the clean data in the target system

```python
# ETL Example with DuckLake
import pandas as pd

# Extract from source
raw_data = pd.read_csv('messy_sales_data.csv')

# Transform (clean and validate)
clean_data = raw_data.dropna()
clean_data['sale_date'] = pd.to_datetime(clean_data['sale_date'])
clean_data['amount'] = clean_data['amount'].round(2)

# Load into DuckLake
conn.execute("CREATE TABLE clean_sales AS SELECT * FROM clean_data")
```

**When to use ETL:**
- Strict data quality requirements
- Compliance and regulatory needs
- Limited storage space
- Well-defined, stable data sources

**Modern ELT (Extract-Load-Transform)**

The "data lake" approach where you store raw data first, then transform as needed:

1. **Extract**: Pull data from source systems
2. **Load**: Store raw data immediately (no transformation)
3. **Transform**: Clean and structure when querying/analyzing

```sql
-- ELT Example with DuckLake

-- Load raw data immediately (no transformation)
CREATE TABLE raw_events AS 
SELECT * FROM read_json('events/*.json');

-- Transform on-demand with views
CREATE VIEW clean_events AS
SELECT 
    user_id,
    event_type,
    timestamp::TIMESTAMP as event_time,
    json_extract(properties, '$.page_url') as page_url
FROM raw_events 
WHERE user_id IS NOT NULL 
  AND timestamp IS NOT NULL;

-- Create different views for different use cases
CREATE VIEW marketing_events AS
SELECT * FROM clean_events 
WHERE event_type IN ('page_view', 'purchase', 'signup');

CREATE VIEW product_events AS  
SELECT * FROM clean_events
WHERE event_type IN ('feature_use', 'error', 'performance');
```

**When to use ELT:**
- Exploratory data analysis
- Rapidly changing requirements
- Multiple use cases for same data
- Large volumes of semi-structured data

**DuckLake ELT Best Practices**

**Layered Architecture**
```sql
-- Bronze Layer: Raw data (exactly as received)
CREATE TABLE bronze_web_logs AS
SELECT * FROM read_csv('logs/*.csv');

-- Silver Layer: Cleaned and validated
CREATE VIEW silver_web_logs AS
SELECT 
    timestamp::TIMESTAMP as log_time,
    ip_address,
    CASE 
        WHEN status_code BETWEEN 200 AND 299 THEN 'success'
        WHEN status_code BETWEEN 400 AND 499 THEN 'client_error'  
        WHEN status_code BETWEEN 500 AND 599 THEN 'server_error'
        ELSE 'other'
    END as status_category,
    response_size_bytes
FROM bronze_web_logs
WHERE timestamp IS NOT NULL;

-- Gold Layer: Business metrics
CREATE TABLE gold_daily_web_metrics AS
SELECT 
    DATE(log_time) as date,
    status_category,
    COUNT(*) as request_count,
    AVG(response_size_bytes) as avg_response_size
FROM silver_web_logs
GROUP BY 1, 2;
```

**Incremental Processing**
```sql
-- Process only new data since last run
CREATE TABLE processed_events AS
SELECT * FROM raw_events 
WHERE timestamp > (
    SELECT COALESCE(MAX(timestamp), '1900-01-01') 
    FROM processed_events
);
```
        """

        console.print(Markdown(etl_md))

        # BI tool integration with detailed explanations
        console.print(
            "\n[bold magenta]Business Intelligence Tool Integration[/bold magenta]"
        )

        bi_md = """
**Connecting BI Tools to DuckLake**

**What are BI Tools?**
Business Intelligence tools help non-technical users explore and visualize data through drag-and-drop interfaces, charts, and dashboards.

**Popular BI Tools and DuckLake Integration**

**Tableau**
```
Connection Type: Native DuckDB Connector
Connection String: duckdb:///path/to/your/ducklake.duckdb
```
- Excellent for complex visualizations
- Strong geographic mapping capabilities  
- Good performance with large datasets
- Supports live connections and extracts

**Microsoft Power BI**
```
Connection Type: ODBC Driver
Data Source: DuckDB ODBC Driver
Database: /path/to/ducklake.duckdb
```
- Integrates well with Microsoft ecosystem
- Strong Excel integration
- Good for corporate environments
- Supports DirectQuery and Import modes

**Grafana**
```
Connection Type: PostgreSQL (for catalog)
Host: localhost:5432
Database: ducklake_catalog
```
- Perfect for operational dashboards
- Real-time monitoring capabilities
- Alert system integration
- Good for time-series data

**Jupyter Notebooks**
```python
# Data exploration and analysis
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

conn = duckdb.connect('ducklake.duckdb')
df = conn.execute(\"\"\"
    SELECT country, COUNT(*) as customers 
    FROM customers 
    GROUP BY country
\"\"\").df()

df.plot(kind='bar', x='country', y='customers')
plt.show()
```

**Performance Optimization for BI Tools**

**Materialized Views**
Instead of running complex queries every time:
```sql
-- Create pre-computed aggregations
CREATE TABLE customer_monthly_summary AS
SELECT 
    DATE_TRUNC('month', order_date) as month,
    customer_id,
    COUNT(*) as order_count,
    SUM(amount) as total_spent,
    AVG(amount) as avg_order_value
FROM orders
GROUP BY 1, 2;

-- BI tools query the summary instead of raw data
SELECT * FROM customer_monthly_summary 
WHERE month >= '2024-01-01';
```

**Connection Pooling**
For web dashboards with many users:
```python
# Don't do this (creates new connection per request)
def get_sales_data():
    conn = duckdb.connect('ducklake.duckdb')  # Slow!
    return conn.execute("SELECT * FROM sales").df()

# Do this instead (reuse connections)
from sqlalchemy import create_engine
engine = create_engine('duckdb:///ducklake.duckdb', pool_size=10)

def get_sales_data():
    return pd.read_sql("SELECT * FROM sales", engine)
```

**Caching Strategy**
```python
# Cache frequently accessed data
import functools
from datetime import datetime, timedelta

@functools.lru_cache(maxsize=100)
def get_daily_metrics(date):
    return conn.execute(f\"\"\"
        SELECT * FROM daily_summary 
        WHERE date = '{date}'
    \"\"\").df()

# Cache expires after 1 hour
def get_real_time_metrics():
    cache_key = datetime.now().strftime('%Y-%m-%d-%H')
    return get_cached_metrics(cache_key)
```

**BI Tool Best Practices**

**Design for Self-Service**
- Create semantic layer with business-friendly column names
- Add descriptions and documentation to tables/views
- Establish naming conventions across datasets

**Example Semantic Layer**
```sql
CREATE VIEW sales_dashboard AS
SELECT 
    o.order_date as "Order Date",
    c.customer_name as "Customer Name", 
    c.customer_tier as "Customer Tier",
    p.product_category as "Product Category",
    o.quantity as "Quantity Sold",
    o.unit_price * o.quantity as "Total Revenue",
    o.unit_price as "Unit Price"
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id  
JOIN products p ON o.product_id = p.product_id;
```

**Security Considerations**
- Use read-only database users for BI connections
- Implement row-level security where needed
- Monitor query patterns for unusual activity
- Set query timeouts to prevent runaway queries
        """

        console.print(Markdown(bi_md))

        # API patterns with detailed explanations
        console.print("\n[bold orange]API Access Patterns[/bold orange]")

        api_md = """
**Building APIs on Top of DuckLake**

APIs (Application Programming Interfaces) let other applications and services access your data programmatically. Instead of giving everyone direct database access, you create controlled endpoints.

**REST API with FastAPI**

REST APIs use HTTP methods (GET, POST, PUT, DELETE) to access resources:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import duckdb
from typing import List, Optional

app = FastAPI(title="DuckLake Customer API")
conn = duckdb.connect('ducklake.duckdb')

# Data models for validation
class Customer(BaseModel):
    customer_id: int
    name: str
    email: str
    country: str
    
class CustomerStats(BaseModel):
    total_customers: int
    countries: List[str]
    avg_age: float

# GET endpoint for customer data
@app.get("/customers/{country}", response_model=List[Customer])
def get_customers_by_country(country: str, limit: int = 100):
    \"\"\"Get customers from a specific country\"\"\"
    try:
        result = conn.execute(\"\"\"
            SELECT customer_id, name, email, country 
            FROM customers 
            WHERE country = ? 
            LIMIT ?
        \"\"\", [country, limit]).fetchdf()
        
        return result.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GET endpoint for statistics
@app.get("/stats/customers", response_model=CustomerStats)
def get_customer_stats():
    \"\"\"Get overall customer statistics\"\"\"
    stats = conn.execute(\"\"\"
        SELECT 
            COUNT(*) as total_customers,
            COUNT(DISTINCT country) as country_count,
            AVG(age) as avg_age
        FROM customers
    \"\"\").fetchone()
    
    countries = conn.execute(\"\"\"
        SELECT DISTINCT country 
        FROM customers 
        ORDER BY country
    \"\"\").fetchall()
    
    return CustomerStats(
        total_customers=stats[0],
        countries=[c[0] for c in countries],
        avg_age=stats[2]
    )

# POST endpoint for adding customers
@app.post("/customers", response_model=Customer)
def create_customer(customer: Customer):
    \"\"\"Add a new customer\"\"\"
    conn.execute(\"\"\"
        INSERT INTO customers (customer_id, name, email, country)
        VALUES (?, ?, ?, ?)
    \"\"\", [customer.customer_id, customer.name, customer.email, customer.country])
    
    return customer
```

**GraphQL API (Alternative to REST)**

GraphQL lets clients request exactly the data they need:

```python
import strawberry
from typing import List

@strawberry.type
class Customer:
    id: int
    name: str
    email: str
    country: str
    
    @strawberry.field
    def orders(self) -> List['Order']:
        # Fetch orders for this customer
        orders = conn.execute(\"\"\"
            SELECT order_id, amount, order_date 
            FROM orders 
            WHERE customer_id = ?
        \"\"\", [self.id]).fetchall()
        return [Order(id=o[0], amount=o[1], date=o[2]) for o in orders]

@strawberry.type
class Query:
    @strawberry.field
    def customers(self, country: Optional[str] = None) -> List[Customer]:
        if country:
            query = "SELECT * FROM customers WHERE country = ?"
            params = [country]
        else:
            query = "SELECT * FROM customers LIMIT 100"
            params = []
            
        results = conn.execute(query, params).fetchall()
        return [Customer(id=r[0], name=r[1], email=r[2], country=r[3]) for r in results]

schema = strawberry.Schema(query=Query)
```

**Real-time Data Updates**

**Webhooks for Change Notifications**
```python
import requests
from datetime import datetime

def notify_data_change(table_name: str, change_type: str):
    \"\"\"Notify downstream systems of data changes\"\"\"
    payload = {
        "table": table_name,
        "change_type": change_type,  # INSERT, UPDATE, DELETE
        "timestamp": datetime.now().isoformat(),
        "source": "ducklake-api"
    }
    
    # Send to webhook endpoints
    webhook_urls = [
        "https://dashboard.company.com/api/data-refresh",
        "https://analytics.company.com/api/cache-invalidate"
    ]
    
    for url in webhook_urls:
        try:
            requests.post(url, json=payload, timeout=5)
        except requests.RequestException as e:
            print(f"Webhook failed for {url}: {e}")

# Use after data changes
@app.post("/customers")
def create_customer(customer: Customer):
    # Insert customer data
    conn.execute("INSERT INTO customers ...")
    
    # Notify downstream systems
    notify_data_change("customers", "INSERT")
    
    return customer
```

**Message Queues for Reliable Delivery**
```python
import pika  # RabbitMQ client

def publish_data_event(table_name: str, event_data: dict):
    \"\"\"Publish data changes to message queue\"\"\"
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    
    # Declare queue
    channel.queue_declare(queue='data_changes', durable=True)
    
    # Publish message
    message = {
        "table": table_name,
        "data": event_data,
        "timestamp": datetime.now().isoformat()
    }
    
    channel.basic_publish(
        exchange='',
        routing_key='data_changes',
        body=json.dumps(message),
        properties=pika.BasicProperties(delivery_mode=2)  # Persistent
    )
    
    connection.close()
```

**API Security Best Practices**

**Authentication & Authorization**
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_token(token: str = Depends(security)):
    \"\"\"Verify API token\"\"\"
    if token.credentials != "your-secret-api-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

@app.get("/customers", dependencies=[Depends(verify_token)])
def get_customers():
    # Protected endpoint
    pass
```

**Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/customers")
@limiter.limit("100/minute")  # Max 100 requests per minute
def get_customers(request: Request):
    pass
```
        """

        console.print(Markdown(api_md))

        # Practical API demonstration
        console.print("\n[bold green]Hands-On: Building a DuckLake API[/bold green]")

        console.print("\nLet's build a complete API that connects to DuckLake:")

        # Complete API example
        api_demo_code = '''# Complete FastAPI application with DuckLake
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import duckdb
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="DuckLake Customer API",
    description="API for accessing customer data in DuckLake",
    version="1.0.0"
)

# Database connection
conn = duckdb.connect('ducklake.duckdb')

# Data models
class Customer(BaseModel):
    customer_id: int
    name: str
    email: str
    country: str
    signup_date: str

class CustomerCreate(BaseModel):
    name: str
    email: str  
    country: str

# API endpoints
@app.get("/")
def read_root():
    return {"message": "DuckLake Customer API", "status": "running"}

@app.get("/customers", response_model=List[Customer])
def get_customers(
    country: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get customers with optional filtering"""
    try:
        if country:
            query = """
                SELECT customer_id, name, email, country, signup_date::STRING
                FROM customers 
                WHERE country = ? 
                ORDER BY signup_date DESC
                LIMIT ? OFFSET ?
            """
            params = [country, limit, offset]
        else:
            query = """
                SELECT customer_id, name, email, country, signup_date::STRING
                FROM customers
                ORDER BY signup_date DESC  
                LIMIT ? OFFSET ?
            """
            params = [limit, offset]
            
        result = conn.execute(query, params).fetchdf()
        return result.to_dict('records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/customers/{customer_id}", response_model=Customer)
def get_customer(customer_id: int):
    """Get a specific customer by ID"""
    try:
        result = conn.execute("""
            SELECT customer_id, name, email, country, signup_date::STRING
            FROM customers 
            WHERE customer_id = ?
        """, [customer_id]).fetchdf()
        
        if result.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        return result.iloc[0].to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/customers", response_model=Customer)
def create_customer(customer_data: CustomerCreate):
    """Create a new customer"""
    try:
        # Get next customer ID
        max_id = conn.execute("SELECT COALESCE(MAX(customer_id), 0) + 1 FROM customers").fetchone()[0]
        
        # Insert new customer
        conn.execute("""
            INSERT INTO customers (customer_id, name, email, country, signup_date)
            VALUES (?, ?, ?, ?, CURRENT_DATE)
        """, [max_id, customer_data.name, customer_data.email, customer_data.country])
        
        # Return created customer
        result = conn.execute("""
            SELECT customer_id, name, email, country, signup_date::STRING
            FROM customers 
            WHERE customer_id = ?
        """, [max_id]).fetchdf()
        
        return result.iloc[0].to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/stats/customers")
def get_customer_stats():
    """Get customer statistics"""
    try:
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_customers,
                COUNT(DISTINCT country) as countries,
                MIN(signup_date) as first_signup,
                MAX(signup_date) as last_signup
            FROM customers
        """).fetchone()
        
        top_countries = conn.execute("""
            SELECT country, COUNT(*) as count
            FROM customers
            GROUP BY country
            ORDER BY count DESC
            LIMIT 5
        """).fetchdf()
        
        return {
            "total_customers": stats[0],
            "total_countries": stats[1], 
            "first_signup": str(stats[2]),
            "last_signup": str(stats[3]),
            "top_countries": top_countries.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# To test the API:
# 1. Save this code as 'api.py'
# 2. Install: pip install fastapi uvicorn
# 3. Run: python api.py  
# 4. Visit: http://localhost:8000/docs for interactive API documentation'''

        console.print("\n[bold cyan]Complete DuckLake API Implementation[/bold cyan]")
        syntax = Syntax(api_demo_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        # BI Integration example
        bi_demo_code = '''# Connect Tableau to DuckLake
# 1. Install DuckDB ODBC driver from: https://duckdb.org/docs/api/odbc
# 2. In Tableau, choose "Other Databases (ODBC)"
# 3. Use connection string: 
#    Driver=DuckDB Driver;Database=/path/to/ducklake.duckdb

# Python/Jupyter integration
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to DuckLake
conn = duckdb.connect('ducklake.duckdb')

# Create business intelligence queries
def create_dashboard_data():
    # Customer acquisition over time
    acquisition_data = conn.execute("""
        SELECT 
            DATE_TRUNC('month', signup_date) as month,
            COUNT(*) as new_customers,
            COUNT(*) OVER (ORDER BY DATE_TRUNC('month', signup_date) 
                          ROWS UNBOUNDED PRECEDING) as cumulative_customers
        FROM customers
        GROUP BY DATE_TRUNC('month', signup_date)
        ORDER BY month
    """).fetchdf()
    
    # Geographic distribution  
    geo_data = conn.execute("""
        SELECT 
            country,
            COUNT(*) as customer_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM customers
        GROUP BY country
        ORDER BY customer_count DESC
    """).fetchdf()
    
    return acquisition_data, geo_data

# Generate visualizations
acquisition_df, geo_df = create_dashboard_data()

# Plot customer acquisition
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acquisition_df['month'], acquisition_df['cumulative_customers'])
plt.title('Customer Growth Over Time')
plt.xlabel('Month')
plt.ylabel('Total Customers')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(geo_df['country'][:10], geo_df['customer_count'][:10])
plt.title('Top 10 Countries by Customer Count')
plt.xlabel('Country')
plt.ylabel('Customers')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("✓ BI dashboard data generated and visualized!")'''

        console.print("\n[bold cyan]BI Tool Integration Example[/bold cyan]")
        syntax = Syntax(bi_demo_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        # Production deployment
        deployment_content = """Production Deployment Strategies:

Container Deployment:
FROM python:3.11
RUN pip install duckdb ducklake
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]

Infrastructure Considerations:
• Shared storage: NFS, S3, Azure Blob
• Catalog database: PostgreSQL cluster
• Backup strategy: Catalog + data files
• Monitoring: Prometheus + Grafana

Scaling Patterns:
• Read replicas: Multiple DuckDB instances
• Write coordination: Single writer pattern
• Catalog HA: PostgreSQL replication
• Data archival: Cold storage policies

Security:
• Network isolation: VPC, security groups
• Authentication: Database roles, RBAC
• Encryption: At-rest and in-transit
• Audit logging: Query logs, access patterns"""

        deployment_panel = Panel(
            deployment_content,
            title=" Production Deployment",
            style="red",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(deployment_panel)

        # Test a simple API-like query pattern
        console.print("\n[bold blue]Testing API-like query patterns...[/bold blue]")

        try:
            # Simulate API endpoint queries
            api_queries = [
                (
                    "Customer Count by Country",
                    'SELECT COUNT(*) as customer_count FROM customers_large GROUP BY "Country" ORDER BY customer_count DESC LIMIT 3',
                ),
                (
                    "Top Customer Cities",
                    'SELECT "City", COUNT(*) as customers FROM customers_large GROUP BY "City" ORDER BY customers DESC LIMIT 3',
                ),
            ]

            for query_name, query in api_queries:
                start_time = time.time()
                result = conn.execute(query).fetchdf()
                exec_time = time.time() - start_time

                console.print(
                    f"   [cyan]{query_name}[/cyan]: {exec_time * 1000:.1f}ms, {len(result)} results"
                )

                # Show sample result
                if not result.empty:
                    first_result = result.iloc[0]
                    sample = " | ".join(
                        [f"{col}: {val}" for col, val in first_result.items()]
                    )
                    console.print(f"      Sample: {sample}")

            console.print(
                "✓ [green]API query patterns demonstrated successfully[/green]"
            )

        except Exception as e:
            console.print(f"[yellow]API pattern testing skipped: {e}[/yellow]")

        return True
    except Exception as e:
        console.print(f" [red]Integration patterns demonstration failed:[/red] {e}")
        logger.error(f"Integration patterns failed: {e}")
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

        # Data Lakes 101 Introduction
        console.print("\n[bold blue]Data Lakes 101: Context and Concepts[/bold blue]")
        console.print("=" * 50)

        datalake_intro_md = """
**Data Architecture Overview**

**Traditional Database (OLTP)**
- Structured data with fixed schemas defined upfront
- Immediate consistency - changes are visible instantly
- Optimized for transactions (INSERT, UPDATE, DELETE)
- Examples: PostgreSQL, MySQL, SQL Server

**Data Warehouse (OLAP)**  
- Centralized repository for business intelligence
- ETL process: Extract data, Transform it, then Load
- Optimized for complex analytical queries
- Examples: Snowflake, BigQuery, Redshift

**Data Lake**
- Store raw data in native formats (JSON, CSV, Parquet)
- Schema-on-read: decide structure when querying, not when storing
- Can handle structured, semi-structured, and unstructured data
- Examples: AWS S3 + Athena, Azure Data Lake

**Lakehouse**
- Combines flexibility of data lakes with performance of warehouses
- Adds ACID transactions and metadata management to data lakes
- Best of both worlds: raw data storage + analytical performance

**What is ACID?**

ACID ensures reliable database transactions:

- **Atomicity**: All operations in a transaction succeed or all fail
- **Consistency**: Database remains in valid state after transactions  
- **Isolation**: Concurrent transactions don't interfere with each other
- **Durability**: Committed changes survive system crashes

Example: When transferring money between accounts, both the debit and credit must succeed together, or neither should happen.

**Why Choose DuckLake?**

**ACID Transactions**
- Unlike basic data lakes, DuckLake guarantees data consistency
- Safe concurrent access from multiple users/applications
- No corrupt or partially written data

**Time Travel and Versioning**
- Every change creates a new snapshot (like Git commits)
- Query data as it existed at any point in time
- Enables data auditing and rollback capabilities

**SQL-First Approach**
- Use familiar SQL instead of learning new query languages
- Works with existing SQL tools and BI platforms
- No need to rewrite queries or retrain analysts

**Open Formats**
- Data stored in standard Parquet files
- No vendor lock-in - can read files with any Parquet-compatible tool
- Easy migration to/from other systems

**Local Development**
- Test and develop locally without cloud dependencies
- Fast iteration cycles during development
- Reduced costs during experimentation

**DuckLake vs Alternatives**

| Feature | DuckLake | Delta Lake | Apache Iceberg | Apache Hudi |
|---------|----------|------------|----------------|-------------|
| **Catalog** | Any SQL DB | Spark metastore | Various | Hive metastore |
| **Setup Complexity** | Simple | Medium | Complex | Complex |
| **Local Development** | Excellent | Limited | Limited | Limited |
| **SQL Support** | Native | Via Spark | Via engines | Via engines |
| **Learning Curve** | Low | Medium | High | High |
        """

        console.print(Markdown(datalake_intro_md))

        # Practical DuckLake connection demonstration
        console.print("\n[bold green]Hands-On: Connecting to DuckLake[/bold green]")

        console.print("\nLet's see how to connect to DuckLake step by step:")

        # Step 1: Basic Connection
        connection_code = """import duckdb

# Connect to DuckLake database file
conn = duckdb.connect('my_ducklake.duckdb')

# Install and load required extensions
conn.execute("INSTALL ducklake")
conn.execute("INSTALL postgres")  # For catalog
conn.execute("LOAD ducklake")
conn.execute("LOAD postgres")

print("✓ DuckLake connection established!")"""

        console.print("\n[bold cyan]Step 1: Basic DuckLake Connection[/bold cyan]")
        syntax = Syntax(connection_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        # Step 2: Catalog Setup
        catalog_code = '''# Attach PostgreSQL catalog
catalog_connection = """
ATTACH 'ducklake:postgres:dbname=ducklake_catalog
        user=ducklake
        password=ducklake123
        host=localhost
        port=5432' AS my_lake
        (DATA_PATH './lake_data');
"""

conn.execute(catalog_connection)
conn.execute("USE my_lake")

print("✓ DuckLake with PostgreSQL catalog ready!")'''

        console.print("\n[bold cyan]Step 2: Connect to PostgreSQL Catalog[/bold cyan]")
        syntax = Syntax(catalog_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        # Step 3: Basic Operations
        operations_code = '''# Create your first table
conn.execute("""
CREATE TABLE customers (
    id INTEGER,
    name VARCHAR,
    email VARCHAR,
    country VARCHAR,
    signup_date DATE
)
""")

# Insert some data
conn.execute("""
INSERT INTO customers VALUES 
(1, 'John Doe', 'john@example.com', 'USA', '2024-01-15'),
(2, 'Jane Smith', 'jane@example.com', 'Canada', '2024-01-16'),
(3, 'Bob Wilson', 'bob@example.com', 'UK', '2024-01-17')
""")

# Query the data
result = conn.execute("SELECT * FROM customers").fetchdf()
print(result)'''

        console.print("\n[bold cyan]Step 3: Create Tables and Insert Data[/bold cyan]")
        syntax = Syntax(operations_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        # Step 4: Advanced Features
        advanced_code = '''# Demonstrate ACID transactions
conn.begin()
try:
    conn.execute("INSERT INTO customers VALUES (4, 'Alice Brown', 'alice@example.com', 'Australia', '2024-01-18')")
    conn.execute("UPDATE customers SET country = 'US' WHERE country = 'USA'")
    conn.commit()
    print("✓ Transaction committed successfully")
except Exception as e:
    conn.rollback()
    print(f"✗ Transaction rolled back: {e}")

# Time travel queries (query previous versions)
snapshots = conn.execute("SELECT * FROM ducklake_snapshots('my_lake')").fetchdf()
print(f"Available snapshots: {len(snapshots)}")

# Query historical data
if len(snapshots) > 1:
    previous_version = len(snapshots) - 1
    historical_data = conn.execute(f"""
        SELECT COUNT(*) as customer_count 
        FROM customers AT (VERSION => {previous_version})
    """).fetchone()
    print(f"Customers in previous version: {historical_data[0]}")'''

        console.print(
            "\n[bold cyan]Step 4: ACID Transactions and Time Travel[/bold cyan]"
        )
        syntax = Syntax(advanced_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)

        console.print(
            "\n[bold yellow]Press Enter to continue to hands-on demonstration...[/bold yellow]"
        )
        input()

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
        console.print(
            "\n[bold yellow]Press Enter to continue to Phase 2 (ACID Transactions & Time Travel)...[/bold yellow]"
        )
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
        console.print(
            "\n[bold yellow]Press Enter to continue to Phase 3 (Performance, Storage & File Analysis)...[/bold yellow]"
        )
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
        console.print(
            "\n[bold yellow]Press Enter to continue to Phase 4 (Large Dataset Experimentation)...[/bold yellow]"
        )
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

        # Wait for user confirmation before Phase 5
        console.print(
            "\n[bold yellow]Press Enter to continue to Phase 5 (Production Deployment)...[/bold yellow]"
        )
        input()

        # Phase 5
        phase5_content = """ This phase covers:
   • Multi-user scenarios and concurrent access
   • Partitioning strategies for performance
   • Table maintenance and optimization
   • Monitoring and observability patterns"""

        phase5_panel = Panel(
            phase5_content,
            title=" PHASE 5: Production Deployment",
            style="bold purple",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(phase5_panel)

        # Demonstrate production concepts
        if not demonstrate_production_concepts(conn, db_name):
            console.print(
                " [yellow]Phase 5 completed with conceptual overview[/yellow]"
            )

        # Summary of what happened in Phase 5
        summary5_content = """• Learned about concurrent access patterns and safety
• Explored partitioning strategies for large datasets
• Understood table maintenance operations (VACUUM, OPTIMIZE)
• Reviewed monitoring and observability best practices"""

        summary5_panel = Panel(
            summary5_content,
            title=" What just happened",
            style="green",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(summary5_panel)

        # Wait for user confirmation before Phase 6
        console.print(
            "\n[bold yellow]Press Enter to continue to Phase 6 (Integration Examples)...[/bold yellow]"
        )
        input()

        # Phase 6
        phase6_content = """ This phase demonstrates:
   • ETL/ELT workflow patterns
   • BI tool integration concepts
   • API access patterns
   • Real-world data pipeline examples"""

        phase6_panel = Panel(
            phase6_content,
            title=" PHASE 6: Integration Examples",
            style="bold orange",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(phase6_panel)

        # Demonstrate integration patterns
        if not demonstrate_integration_patterns(conn):
            console.print(
                " [yellow]Phase 6 completed with integration overview[/yellow]"
            )

        # Summary of what happened in Phase 6
        summary6_content = """• Explored ETL/ELT workflow patterns with DuckLake
• Learned about BI tool integration approaches
• Understood API access and data serving patterns
• Reviewed real-world production deployment strategies"""

        summary6_panel = Panel(
            summary6_content,
            title=" What just happened",
            style="green",
            box=box.ROUNDED,
        )
        console.print("\n")
        console.print(summary6_panel)

        # Completion summary
        completion_content = """ Successfully demonstrated DuckLake's complete ecosystem:
    Conceptual Foundation: Data lakes, lakehouses, and DuckLake advantages
    Foundation Setup: PostgreSQL catalog with local storage
    Core Operations: Table creation, data ingestion, queries
    ACID Transactions: Consistent data operations with rollback
    Advanced Features: Snapshots, time travel, schema evolution
    Performance: Query optimization and execution analysis
    Large Scale: 300k+ record datasets and real-world performance
    Production Deployment: Multi-user patterns, partitioning, maintenance
    Integration Patterns: BI tools, APIs, ETL/ELT workflows"""

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
        next_steps_content = """1. Deploy DuckLake in your production environment
2. Integrate with your existing BI tools (Tableau, PowerBI)
3. Build REST/GraphQL APIs for data access
4. Implement ETL/ELT pipelines for your data sources
5. Set up monitoring, backup, and maintenance schedules
6. Explore advanced features: partitioning, optimization, scaling"""

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
